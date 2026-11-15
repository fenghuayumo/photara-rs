#!/usr/bin/env python3
"""Package the Blender add-on into a zip ready to install in Blender."""

import os
import sys
import zipfile

from version import check_version


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ADDON_DIR = os.path.join(ROOT, "photara_rs_engine")
DIST_DIR = os.path.join(ROOT, "dist")


def addon_version():
    return check_version()


def main():
    version = addon_version()
    os.makedirs(DIST_DIR, exist_ok=True)
    output = os.path.join(DIST_DIR, f"photara_blender_addon-{version}.zip")

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(os.listdir(ADDON_DIR)):
            if not name.endswith(".py") and not name.endswith(".md"):
                continue
            path = os.path.join(ADDON_DIR, name)
            archive.write(path, arcname=f"photara_rs_engine/{name}")

    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
