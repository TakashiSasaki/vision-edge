"""Run the approved PR-05 checks; retain output, exits and exact source hashes.

Usage: python3 docs/verification/pr05-f01/verify.py
Local browser override and writable npm cache are explicit environment choices.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess

root = Path(__file__).resolve().parents[3]
logs = root / "docs/verification/pr05-f01/logs"
logs.mkdir(parents=True, exist_ok=True)
commands = [
    ["node", "--version"], ["npm", "--version"], ["npm", "ci"],
    ["npm", "run", "typecheck"], ["npm", "run", "lint"], ["npm", "test"],
    ["npm", "run", "build"], ["npm", "run", "test:e2e"],
    ["npm", "run", "test:e2e:controls"], ["git", "diff", "--check"],
]
env = dict(os.environ, npm_config_cache="/workspace/.npm-cache",
           PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH="/usr/bin/chromium")
files = sorted({p for folder in ["src", "e2e", "scripts", ".github"]
                for p in (root / folder).rglob("*") if p.is_file()}
               | {root / name for name in ["package.json", "package-lock.json", ".nvmrc",
                  "vite.config.ts", "vitest.config.ts", "playwright.config.ts", "tsconfig.json", "index.html"]})
binding = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
           "tree": subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True).strip(),
           "sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(logs / "validated-source.json").write_text(json.dumps(binding, indent=2) + "\n")
checks = []
for index, command in enumerate(commands):
    start = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    name = f"final-{index:02}.txt"
    (logs / name).write_bytes(result.stdout)
    checks.append({"command": command, "exit": result.returncode, "start": start,
                   "end": datetime.now(timezone.utc).isoformat(), "log": name,
                   "sha256": hashlib.sha256(result.stdout).hexdigest()})
    (logs / "checks.json").write_text(json.dumps({"environment": {
        "npm_config_cache": env["npm_config_cache"],
        "PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH": env["PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH"]},
        "checks": checks}, indent=2) + "\n")
    print(f'{" ".join(command)}: exit {result.returncode} ({name})', flush=True)
    if result.returncode:
        print(result.stdout.decode(errors="replace")[-6000:], flush=True)
        raise SystemExit(result.returncode)
