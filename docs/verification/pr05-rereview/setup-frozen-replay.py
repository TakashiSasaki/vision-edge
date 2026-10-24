"""Extract frozen PR12 probes without touching PR12 or reviewed product files."""
import io
from pathlib import Path
import subprocess
import sys
import tarfile

root=Path(__file__).resolve().parents[3]
destination=Path(sys.argv[1]).resolve()
destination.mkdir()  # Refuse to overwrite an existing replay directory.
archive=subprocess.check_output(['git','archive','57ee68bc503cfc1633039d948d1b39fcf341caa5','docs/verification/pr05-independent'],cwd=root)
with tarfile.open(fileobj=io.BytesIO(archive))as tar:tar.extractall(destination,filter='data')
(destination/'src').symlink_to(root/'src',target_is_directory=True)
(destination/'node_modules').symlink_to(root/'node_modules',target_is_directory=True)
print(destination)
