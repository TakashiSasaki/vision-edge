"""Reproduce the final coordinate vectors against unmodified PR-04 product code.

Run from the author checkout after npm ci. Exit 1 is the preserved test failure,
not an accepted success. Supply an unused writable path as the only argument.
"""
from pathlib import Path
import shutil
import subprocess
import sys
import os

root = Path(__file__).resolve().parents[3]
target = Path(sys.argv[1]).resolve()
base = "f97e19ac5f9f07eca23732003844b5082cd27d37"
subprocess.run(["git", "worktree", "add", "--detach", str(target), base], cwd=root, check=True)
# Dependency/lock files are unchanged; this replay reuses the locked author install.
(target / "node_modules").symlink_to(root / "node_modules", target_is_directory=True)
shutil.copyfile(root / "e2e/coordinates.spec.ts", target / "e2e/coordinates.spec.ts")
env = dict(os.environ, PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH="/usr/bin/chromium",
           npm_config_cache="/workspace/.npm-cache")
subprocess.run(["npm", "run", "build"], cwd=target, env=env, check=True)
result = subprocess.run(["npm", "run", "test:e2e", "--", "e2e/coordinates.spec.ts"], cwd=target, env=env)
raise SystemExit(result.returncode)
