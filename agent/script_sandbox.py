"""Sandboxed execution for LLM-generated audit scripts.

The auditor executes scripts written by an LLM prompted with untrusted
paper text — a prompt-injected paper can smuggle code, so generated code
never runs unsandboxed on the host.

Policy, in order:

1. macOS `sandbox-exec` — deny-by-default profile: no network, no exec
   outside the interpreter, writes only under the workdir, home
   directories unreadable, rlimits + a minimal env.
2. Linux `bwrap` (bubblewrap) — private namespaces (no network),
   read-only system binds, the workdir is the only writable mount,
   same minimal env + rlimits.
3. Otherwise generated code does not run at all: attempts record
   `sandbox_unavailable`. `LEMMA_AUDIT_HOST_EXEC=1` is an explicit
   operator opt-in for platforms with no sandbox — still minimal env
   (no inherited secrets) but no filesystem/network isolation. Plugin
   code must never set it.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
BWRAP = "bwrap"
HOST_EXEC_ENV = "LEMMA_AUDIT_HOST_EXEC"

# Generated scripts are CPU-numpy by contract; caps bound a runaway.
ADDRESS_SPACE_BYTES = 4 << 30
FILE_BYTES = 64 << 20
NOFILE = 256

_LAUNCHER = """\
import resource, runpy, sys
def cap(res, want):
    soft, hard = resource.getrlimit(res)
    target = min(want, hard) if hard > 0 else want
    resource.setrlimit(res, (target, hard))
cap(resource.RLIMIT_CPU, {cpu})
try:
    cap(resource.RLIMIT_AS, {addr})
except (ValueError, OSError):
    pass
cap(resource.RLIMIT_FSIZE, {fsize})
cap(resource.RLIMIT_NOFILE, {nofile})
runpy.run_path(sys.argv[1], run_name='__main__')
"""


def _launcher(timeout_s: float) -> str:
    return _LAUNCHER.format(
        cpu=int(timeout_s) + 60,
        addr=ADDRESS_SPACE_BYTES,
        fsize=FILE_BYTES,
        nofile=NOFILE,
    )


def _minimal_env(work: Path) -> dict[str, str]:
    w = str(work.resolve())
    return {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": f"{w}/home",
        "TMPDIR": f"{w}/tmp",
        "MPLCONFIGDIR": f"{w}/mplconfig",
        "LANG": "C.UTF-8",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1",
    }


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


def _macos_profile(work: Path) -> str:
    execs = _exec_literals()
    reopen = [
        _sub(work.resolve()),
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


def _bwrap_cmd(
    script: Path, work: Path, env: dict[str, str], timeout_s: float
) -> list[str]:
    cmd = [
        BWRAP,
        "--die-with-parent",
        "--unshare-all",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    # Minimal read-only root: interpreter + libs + fonts. Home dirs and
    # the rest of the filesystem simply do not exist in the namespace.
    for d in (
        "/usr",
        "/lib",
        "/lib64",
        "/bin",
        "/sbin",
        "/etc/fonts",
        "/etc/ld.so.cache",
        "/etc/resolv.conf",
    ):
        if Path(d).exists():
            cmd += ["--ro-bind", d, d]
    for p in {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}:
        if p.is_dir():
            cmd += ["--ro-bind", str(p), str(p)]
    # Parent dirs for the workdir mount must exist in the namespace.
    resolved = work.resolve()
    for parent in resolved.parents:
        cmd += ["--dir", str(parent)]
    cmd += ["--bind", str(resolved), str(resolved), "--chdir", str(resolved)]
    cmd.append("--clearenv")
    for k, v in env.items():
        cmd += ["--setenv", k, v]
    cmd += [sys.executable, "-I", "-c", _launcher(timeout_s), str(script.resolve())]
    return cmd


def sandbox_kind() -> str | None:
    if platform.system() == "Darwin" and Path(SANDBOX_EXEC).exists():
        return "sandbox-exec"
    if platform.system() == "Linux" and shutil.which(BWRAP):
        return "bwrap"
    return None


class ScriptRun:
    def __init__(self) -> None:
        self.exit_code = -1
        self.stdout = ""
        self.stderr = ""
        self.wall_s = 0.0
        self.mode = "none"
        self.note: str | None = None


def _spawn(argv: list[str], work: Path, env: dict, timeout_s: float) -> ScriptRun:
    res = ScriptRun()
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            argv,
            cwd=work,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            out, err = proc.communicate(timeout=timeout_s)
            res.exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            out, err = proc.communicate()
            res.exit_code = -1
            err = (err or "") + f"\nTIMEOUT after {timeout_s}s"
        res.stdout, res.stderr = out or "", err or ""
    except OSError as e:
        res.stderr = f"sandbox spawn failed: {e}"
    res.wall_s = time.time() - t0
    return res


def run_generated(script_path: Path, workdir: Path, timeout_s: float) -> ScriptRun:
    """Run a generated script under the best available isolation.

    Never falls back to unsandboxed execution unless the operator set
    LEMMA_AUDIT_HOST_EXEC=1 — a missing sandbox means the attempt
    records sandbox_unavailable, not a host run.
    """
    res = ScriptRun()
    work = workdir.resolve()
    script = script_path.resolve()
    env = _minimal_env(work)
    for d in ("home", "tmp", "mplconfig"):
        (work / d).mkdir(parents=True, exist_ok=True)

    kind = sandbox_kind()
    argv = None
    if kind == "sandbox-exec":
        res.mode = "sandbox-exec"
        argv = [
            SANDBOX_EXEC,
            "-p",
            _macos_profile(work),
            sys.executable,
            "-I",
            "-c",
            _launcher(timeout_s),
            str(script),
        ]
    elif kind == "bwrap":
        res.mode = "bwrap"
        argv = _bwrap_cmd(script, work, env, timeout_s)
    elif os.environ.get(HOST_EXEC_ENV) == "1":
        res.mode = "host-exec"
        res.note = "LEMMA_AUDIT_HOST_EXEC=1 — generated code ran without isolation"
        argv = [
            sys.executable,
            "-I",
            "-c",
            _launcher(timeout_s),
            str(script),
        ]
    if argv is not None:
        run = _spawn(argv, work, env, timeout_s)
        run.mode, run.note = res.mode, res.note
        return run

    res.note = (
        "sandbox_unavailable: no sandbox-exec (macOS) or bwrap (Linux) on "
        f"this host; generated code was NOT executed. To run without "
        f"isolation (unsafe), set {HOST_EXEC_ENV}=1."
    )
    res.stderr = res.note
    return res
