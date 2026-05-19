"""Mirror plot directories to /eos/user/m/mdefranc/www/WW_threshold/<subdir>/
and drop an index.php in every directory so they render as a browsable
web gallery.

The publishing pattern (eos_publish.py + index.php) is shared with the
WW kinematic-reconstruction project (private/WW/WW_reco/); we use a
separate EOS parent directory to keep the threshold-fit output isolated.

Override via env vars EOS_BASE_DIR / EOS_BASE_URL for staging or for a
colleague's web area."""

import os
import shutil
import subprocess
from datetime import date

EOS_BASE      = os.environ.get("EOS_BASE_DIR",
                                "/eos/user/m/mdefranc/www/WW_threshold")
EOS_URL       = os.environ.get("EOS_BASE_URL",
                                "https://mdefranc.web.cern.ch/WW_threshold")
INDEX_PHP_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.php")
EXTS          = (".png", ".pdf")


def _drop_index(directory):
    if not os.path.isfile(INDEX_PHP_SRC):
        return
    shutil.copy2(INDEX_PHP_SRC, os.path.join(directory, "index.php"))


def publish(src_dir, subdir):
    """Mirror src_dir into EOS_BASE/subdir (PNG+PDF only) and place index.php
    in every directory along the way, including all parents up to EOS_BASE."""
    if not os.path.isdir(src_dir):
        print(f"[publish] SKIP: source {src_dir} not found")
        return

    dst_root = os.path.join(EOS_BASE, subdir)
    n_copied = 0
    for root, _, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        out = dst_root if rel == "." else os.path.join(dst_root, rel)
        os.makedirs(out, exist_ok=True)
        for f in files:
            if f.lower().endswith(EXTS):
                shutil.copy2(os.path.join(root, f), os.path.join(out, f))
                n_copied += 1
        _drop_index(out)

    p = os.path.dirname(dst_root)
    while p.startswith(EOS_BASE) and p != EOS_BASE:
        os.makedirs(p, exist_ok=True)
        _drop_index(p)
        p = os.path.dirname(p)
    os.makedirs(EOS_BASE, exist_ok=True)
    _drop_index(EOS_BASE)

    print(f"[publish] copied {n_copied} files  →  {dst_root}")
    print(f"[publish] URL: {EOS_URL}/{subdir}/")


def archive_tag():
    """`<YYYY-MM-DD>_<short-sha>`; appends `-dirty` if the working tree has uncommitted changes."""
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"],
                           capture_output=True, text=True, check=True).stdout.strip()
    suffix = "-dirty" if dirty else ""
    return f"{date.today().isoformat()}_{sha}{suffix}"
