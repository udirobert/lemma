from __future__ import annotations

import contextlib
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from agent.room_manifest import RUN_LIMITS

SANDBOX_EXEC = "/usr/bin/sandbox-exec"

_LAUNCHER = f"""\
import resource, runpy, sys
resource.setrlimit(resource.RLIMIT_CPU, ({RUN_LIMITS["cpu_s"]}, {RUN_LIMITS["cpu_s"]}))
resource.setrlimit(resource.RLIMIT_AS, ({RUN_LIMITS["address_space_bytes"]}, {RUN_LIMITS["address_space_bytes"]}))
resource.setrlimit(resource.RLIMIT_FSIZE, ({RUN_LIMITS["file_bytes"]}, {RUN_LIMITS["file_bytes"]}))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
runpy.run_path(sys.argv[1], run_name='__main__')
"""


def _sub(path: Path | str) -> str:
    return f"(subpath {json.dumps(str(path))})"


def _lit(path: Path | str) -> str:
    return f"(literal {json.dumps(str(path))})"


def _exec_literals() -> list[str]:
    python = Path(sys.executable)
    resolved = python.resolve()
    literals = {_lit(python), _lit(resolved)}
    for candidate in resolved.parent.parent.glob(
        "Resources/Python.app/Contents/MacOS/Python"
    ):
        literals.add(_lit(candidate))
        literals.add(_lit(candidate.resolve()))
    return sorted(literals)


def build_profile(work: Path, snapshot: Path) -> str:
    execs = _exec_literals()
    reopen = [
        _sub(work.resolve()),
        _sub(snapshot.resolve()),
        _sub(Path(sys.prefix).resolve()),
        _sub(Path(sys.base_prefix).resolve()),
    ]
    deny = " ".join(
        f"(deny file-read-data file-read-xattr {_sub(p)})"
        for p in ("/Users", "/home", "/Volumes")
    )
    return "\n".join(
        [
            "(version 1)",
            "(deny default)",
            f"(allow process-exec {' '.join(execs)})",
            "(allow file-read-metadata)",
            "(allow file-read*)",
            deny,
            f"(allow file-read-data file-read-xattr {' '.join(reopen)})",
            f"(allow file-write* {_sub(work.resolve())} {_lit('/dev/null')})",
            "(allow sysctl-read)",
            "(allow mach-lookup)",
            "",
        ]
    )


def sandbox_env(work: Path) -> dict[str, str]:
    w = str(work.resolve())
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": f"{w}/home",
        "TMPDIR": f"{w}/tmp",
        "MPLCONFIGDIR": f"{w}/mplconfig",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1",
    }


def supported() -> tuple[bool, str | None]:
    if platform.system() != "Darwin":
        return False, "sandbox-exec requires macOS"
    if not Path(SANDBOX_EXEC).exists():
        return False, f"{SANDBOX_EXEC} not found"
    if not hasattr(resource, "RLIMIT_AS"):
        return False, "RLIMIT_AS unsupported on this platform"
    return True, None


class SandboxResult:
    def __init__(self) -> None:
        self.exit_code: int | None = None
        self.timed_out = False
        self.wall_s: float = 0.0
        self.stdout: bytes = b""
        self.stderr: bytes = b""
        self.stdout_truncated = False
        self.stderr_truncated = False
        self.error: str | None = None


def _read_bounded(path: Path) -> tuple[bytes, bool]:
    limit = RUN_LIMITS["max_output_bytes"]
    data = path.read_bytes() if path.is_file() else b""
    if len(data) > limit:
        return data[:limit], True
    return data, False


def run_script(script_path: Path, work: Path, out_dir: Path) -> SandboxResult:
    ok, reason = supported()
    result = SandboxResult()
    if not ok:
        result.error = f"isolation unavailable: {reason}"
        return result
    work = work.resolve()
    out_dir = out_dir.resolve()
    for d in ("home", "tmp", "mplconfig"):
        (work / d).mkdir(parents=True, exist_ok=True)
    (work / "results" / "c6").mkdir(parents=True, exist_ok=True)
    snapshot = script_path.resolve().parent
    profile = build_profile(work, snapshot)
    out_path = out_dir / "stdout.txt"
    err_path = out_dir / "stderr.txt"
    start = time.monotonic()
    with open(out_path, "wb") as stdout_f, open(err_path, "wb") as stderr_f:
        proc = subprocess.Popen(
            [
                SANDBOX_EXEC,
                "-p",
                profile,
                sys.executable,
                "-I",
                "-c",
                _LAUNCHER,
                str(script_path.resolve()),
            ],
            cwd=work,
            env=sandbox_env(work),
            stdout=stdout_f,
            stderr=stderr_f,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            result.exit_code = proc.wait(timeout=RUN_LIMITS["wall_s"])
        except subprocess.TimeoutExpired:
            result.timed_out = True
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            result.exit_code = proc.wait()
    result.wall_s = time.monotonic() - start
    result.stdout, result.stdout_truncated = _read_bounded(out_path)
    result.stderr, result.stderr_truncated = _read_bounded(err_path)
    return result


_PROBE_CHILD = """\
import os, socket, subprocess, tempfile
checks = {}
sentinel = SENTINEL_PATH
try:
    open(sentinel).read()
    checks['read_denied'] = False
except Exception:
    checks['read_denied'] = True
try:
    open('/tmp/lemma_probe_forbidden_write.txt', 'w').write('x')
    checks['write_denied'] = False
except Exception:
    checks['write_denied'] = True
try:
    s = socket.socket()
    s.settimeout(2)
    s.connect(('127.0.0.1', 9))
    checks['net_denied'] = False
except Exception:
    checks['net_denied'] = True
try:
    subprocess.run(['/usr/bin/true'])
    checks['exec_denied'] = False
except Exception:
    checks['exec_denied'] = True
checks['no_secret'] = 'LEMMA_PROBE_SECRET' not in os.environ
try:
    import numpy as np
    checks['numpy'] = float(np.arange(8).sum()) == 28.0
except Exception:
    checks['numpy'] = False
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.savefig(os.path.join(tempfile.gettempdir(), 'probe.png'))
    checks['matplotlib'] = True
except Exception:
    checks['matplotlib'] = False
print('PROBE_JSON:' + __import__('json').dumps(checks))
"""


def readiness_probe() -> dict:
    ok, reason = supported()
    if not ok:
        return {"available": False, "reason": reason, "checks": {}}
    base = Path(tempfile.mkdtemp(prefix="lemma-probe-"))
    work = base / "work"
    snap = base / "snapshot"
    out_dir = base / "out"
    sentinel = base / "sentinel.txt"
    for d in (work, snap, out_dir):
        d.mkdir(parents=True)
    sentinel.write_text("lemma-probe-sentinel", encoding="utf-8")
    probe = snap / "probe.py"
    probe.write_text(
        _PROBE_CHILD.replace("SENTINEL_PATH", json.dumps(str(sentinel))),
        encoding="utf-8",
    )
    result = SandboxResult()
    profile = build_profile(work, snap)
    out_path = out_dir / "stdout.txt"
    err_path = out_dir / "stderr.txt"
    env = sandbox_env(work)
    os.environ["LEMMA_PROBE_SECRET"] = "sentinel-value"
    try:
        with open(out_path, "wb") as so, open(err_path, "wb") as se:
            proc = subprocess.Popen(
                [
                    SANDBOX_EXEC,
                    "-p",
                    profile,
                    sys.executable,
                    "-I",
                    "-c",
                    _LAUNCHER,
                    str(probe),
                ],
                cwd=work,
                env=env,
                stdout=so,
                stderr=se,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=RUN_LIMITS["wall_s"])
            except subprocess.TimeoutExpired:
                result.timed_out = True
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    finally:
        del os.environ["LEMMA_PROBE_SECRET"]
    result.stdout, _ = _read_bounded(out_path)
    result.stderr, _ = _read_bounded(err_path)
    checks: dict[str, bool] = {}
    for line in result.stdout.decode("utf-8", "replace").splitlines():
        if line.startswith("PROBE_JSON:"):
            try:
                checks = json.loads(line[len("PROBE_JSON:") :])
            except json.JSONDecodeError:
                checks = {}
    shutil.rmtree(base, ignore_errors=True)
    required = [
        "read_denied",
        "write_denied",
        "net_denied",
        "exec_denied",
        "no_secret",
        "numpy",
        "matplotlib",
    ]
    failed = [k for k in required if not checks.get(k)]
    if not checks:
        return {
            "available": False,
            "reason": "probe produced no result",
            "checks": {},
            "stderr_tail": result.stderr.decode("utf-8", "replace")[-500:],
        }
    if failed:
        return {
            "available": False,
            "reason": f"probe checks failed: {', '.join(failed)}",
            "checks": checks,
        }
    return {"available": True, "reason": None, "checks": checks}
