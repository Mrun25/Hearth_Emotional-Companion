# -*- coding: utf-8 -*-
"""
setup_startup.py -- One-Time Windows Startup Registration
=========================================================
Copies startup_runner.bat into the Windows Startup folder so that the
Fumii evaluation pipeline runs automatically on every boot.

Run this ONCE after cloning/setting up the project:
    python pipeline/setup_startup.py

What it does:
  • Locates the Windows user Startup folder via the APPDATA environment variable
  • Copies pipeline/startup_runner.bat into that folder
  • Prints a confirmation message

To undo (disable auto-start):
    Delete the file from: %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def get_startup_folder() -> Path:
    """Return the path to the current user's Windows Startup folder."""
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise EnvironmentError(
            "APPDATA environment variable is not set. "
            "This script must be run on Windows."
        )
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def main() -> None:
    # Locate startup_runner.bat relative to this script
    pipeline_dir = Path(__file__).resolve().parent
    bat_src = pipeline_dir / "startup_runner.bat"

    if not bat_src.exists():
        print(f"[ERROR] startup_runner.bat not found at: {bat_src}")
        print("        Make sure you run this script from inside the project directory.")
        sys.exit(1)

    # Get the Windows Startup folder
    try:
        startup_folder = get_startup_folder()
    except EnvironmentError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if not startup_folder.exists():
        print(f"[ERROR] Startup folder does not exist: {startup_folder}")
        print("        This is unexpected on a normal Windows installation.")
        sys.exit(1)

    bat_dst = startup_folder / "fumii_eval_pipeline.bat"

    # Copy the file
    try:
        shutil.copy2(bat_src, bat_dst)
    except PermissionError:
        print(f"[ERROR] Permission denied copying to: {bat_dst}")
        print("        Try running this script as Administrator.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Fumii Eval Pipeline — Startup Registration Complete")
    print("=" * 60)
    print(f"\n  ✓ Installed: {bat_dst}")
    print(f"\n  The pipeline will now run automatically every time")
    print(f"  you log into Windows.")
    print(f"\n  Log file location:")
    print(f"    {pipeline_dir / 'fumii_eval_log.txt'}")
    print(f"\n  To disable auto-start, delete:")
    print(f"    {bat_dst}")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
