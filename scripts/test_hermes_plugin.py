"""Install-smoke test for the Hermes plugin's subdir-install contract.

The catalog installs `subdir: hermes-plugin`, so only that directory is
checked out — the pipeline must live inside it (hermes-plugin/
lemma_pipeline/, vendored from agent/ by scripts/sync_hermes_plugin.py).
This test copies hermes-plugin/ alone to a temp dir (exactly what an
install sees) and verifies resolution, CLI launch, config bridging, and
the doctor preflight — the same checks a catalog reviewer runs by hand.

Run: python scripts/test_hermes_plugin.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLUGIN = REPO / "hermes-plugin"


class SubdirInstallTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="lemma-plugin-"))
        cls.install = cls.tmp / "hermes-plugin"
        shutil.copytree(
            PLUGIN, cls.install, ignore=shutil.ignore_patterns("__pycache__")
        )
        cls.env = {
            **os.environ,
            "LEMMA_PLUGIN_DATA": str(cls.tmp / "data"),
            "PYTHONPATH": str(cls.install),
        }

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _call(self, expr: str) -> str:
        proc = subprocess.run(
            [sys.executable, "-c", expr],
            cwd=self.install,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        return proc.stdout.strip()

    def test_pipeline_resolves_inside_plugin_dir(self):
        out = self._call(
            "import sys; sys.path.insert(0, '.'); import tools; "
            "print(tools._pipeline_root())"
        )
        self.assertEqual(Path(out).resolve(), self.install.resolve())

    def test_cli_launches_vendored_package(self):
        proc = subprocess.run(
            [sys.executable, "-m", "lemma_pipeline.cli", "judge", "--help"],
            cwd=self.install,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        self.assertIn("judge", proc.stdout)

    def test_plugin_config_bridges_to_child_env(self):
        out = self._call(
            "import sys; sys.path.insert(0, '.')\n"
            "import tools\n"
            "class Ctx:\n"
            "    def get_config(self, k, default=None):\n"
            "        return {'host_exec': True, 'weave': 'yes'}.get(k, default)\n"
            "env = tools._child_env(Ctx())\n"
            "print(env.get('LEMMA_AUDIT_HOST_EXEC'), env.get('LEMMA_WEAVE'))"
        )
        self.assertEqual(out, "1 1")

    def test_child_env_propagates_site_packages(self):
        """PM puts the dependency env's site-packages on the host's
        sys.path; children spawned from sys.executable don't inherit
        sys.path, so _child_env must pass them via PYTHONPATH —
        otherwise plugin python_dependencies are invisible to the
        pipeline (regression: fitz ModuleNotFoundError on install)."""
        out = self._call(
            "import sys; sys.path.insert(0, '.')\n"
            "import tools\n"
            "site = [p for p in sys.path if 'site-packages' in p]\n"
            "env = tools._child_env()\n"
            "pp = env.get('PYTHONPATH', '')\n"
            "print('yes' if site and all(p in pp for p in site) else 'no')"
        )
        self.assertEqual(out, "yes")

    def test_doctor_returns_json_checks(self):
        out = self._call(
            "import sys; sys.path.insert(0, '.'); import tools, json; "
            "d = json.loads(tools.lemma_doctor({})); "
            "print(json.dumps({'ok': d['ok'], 'checks': sorted(d['checks'])}))"
        )
        payload = json.loads(out)
        self.assertIn("pipeline", payload["checks"])
        self.assertIn("sandbox", payload["checks"])
        self.assertIn("trackio", payload["checks"])
        # deps/trackio/key may legitimately be absent outside a Hermes env —
        # the contract is that doctor reports them, not that they pass here.
        self.assertIsInstance(payload["ok"], bool)

    def test_no_agent_package_shadowing(self):
        """The vendored package must not be named `agent` — that would
        collide with hermes-agent's own agent package on sys.path."""
        self.assertFalse((self.install / "agent").exists())
        self.assertTrue((self.install / "lemma_pipeline" / "cli.py").is_file())

    def test_no_stale_agent_cli_invocations(self):
        """Every pipeline spawn must use lemma_pipeline.cli — a stray
        `agent.cli` literal fails only on catalog installs (regression:
        lemma_audit/improve Popen calls missed the rename once)."""
        src = (self.install / "tools.py").read_text(encoding="utf-8")
        self.assertNotIn('"agent.cli"', src)
        self.assertNotIn("-m agent.cli", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
