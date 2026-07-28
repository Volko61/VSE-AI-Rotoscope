#!/usr/bin/env python3
"""Build the release ZIPs for extensions.blender.org and check them.

Wraps `blender --command extension build --split-platforms` so that:

  * output always lands in ./dist/, never inside the package directory (a ZIP
    left next to blender_manifest.toml gets swallowed by the *next* build),
  * dist/ is emptied first, so stale versions are never uploaded by mistake,
  * every produced ZIP is verified: under the platform's 200 MB cap, and free
    of __pycache__ / nested ZIPs / stray junk.

Usage:
    python scripts/build.py                     # auto-detect blender
    python scripts/build.py --blender "C:/.../blender.exe"
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "auto_rotoscope")
DIST = os.path.join(ROOT, "dist")

# extensions.blender.org rejects uploads above this (HTTP 413).
SIZE_CAP = 200 * 1024 * 1024

# Anything matching these must not appear in a release ZIP.
FORBIDDEN = ("__pycache__/", ".pyc", ".part", ".zip", ".git")

# Files the add-on cannot run without.
REQUIRED = (
    "blender_manifest.toml",
    "__init__.py",
    "models/sam2.1_hiera_tiny/encoder.onnx",
    "models/sam2.1_hiera_tiny/decoder.onnx",
    "licenses/NOTICE.txt",
    "licenses/GPL-3.0.txt",
)


def find_blender(explicit: str | None) -> str:
    if explicit:
        return explicit
    if (env := os.environ.get("BLENDER")):
        return env
    if (which := shutil.which("blender")):
        return which
    # Newest first, so a 5.x install wins over an older one.
    candidates = sorted(
        glob.glob("C:/Program Files/Blender Foundation/Blender */blender.exe")
        + glob.glob("/usr/share/blender/*/blender")
        + glob.glob("/opt/blender*/blender"),
        reverse=True,
    )
    if candidates:
        return candidates[0]
    sys.exit("Could not find Blender. Pass --blender /path/to/blender")


def run(blender: str, *args: str) -> None:
    cmd = [blender, "--command", "extension", *args]
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=PKG, check=True)


def check(path: str) -> bool:
    """Verify one built ZIP. Returns True if it passes."""
    name = os.path.basename(path)
    size = os.path.getsize(path)
    ok = True

    pct = size * 100 // SIZE_CAP
    verdict = "OK" if size <= SIZE_CAP else "TOO LARGE"
    print(f"\n{name}\n  {size / 2**20:.1f} MiB  ({pct}% of the 200 MB cap) - {verdict}")
    if size > SIZE_CAP:
        print("  !! extensions.blender.org will reject this upload with HTTP 413.")
        ok = False

    with zipfile.ZipFile(path) as zf:
        # Blender writes the package contents at the ZIP root (no directory
        # prefix), so REQUIRED paths compare directly against the entry names.
        names = zf.namelist()
        entries = set(names)

        junk = [n for n in names if any(f in n for f in FORBIDDEN)]
        if junk:
            ok = False
            print("  !! forbidden entries:")
            for n in junk[:10]:
                print(f"       {n}")

        missing = [r for r in REQUIRED if r not in entries]
        if missing:
            ok = False
            print("  !! missing required files:")
            for m in missing:
                print(f"       {m}")

        wheels = sorted(n for n in names if n.endswith(".whl"))
        print(f"  wheels ({len(wheels)}):")
        for w in wheels:
            print(f"       {os.path.basename(w)}")

    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blender", help="path to the Blender executable")
    args = ap.parse_args()
    blender = find_blender(args.blender)

    for rel in ("models/sam2.1_hiera_tiny/encoder.onnx", "wheels"):
        p = os.path.join(PKG, rel)
        if not os.path.exists(p) or (os.path.isdir(p) and not os.listdir(p)):
            sys.exit(f"Missing {rel}. Run `python scripts/fetch_assets.py` first.")

    if os.path.isdir(DIST):
        shutil.rmtree(DIST)
    os.makedirs(DIST)

    run(blender, "validate")
    run(blender, "build", "--split-platforms", "--output-dir", DIST)

    zips = sorted(glob.glob(os.path.join(DIST, "*.zip")))
    if not zips:
        sys.exit("Build produced no ZIP files.")

    print("\n" + "=" * 68)
    all_ok = all([check(z) for z in zips])
    print("=" * 68)

    if all_ok:
        print(f"\nAll {len(zips)} package(s) passed. Upload from: {DIST}")
        print("Next: https://extensions.blender.org/submit/  (see RELEASE.md)")
        return 0
    print("\nSome packages failed the checks above — do not upload them.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
