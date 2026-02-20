#!/usr/bin/env python3
import subprocess
import sys

cmd = [
    sys.executable,
    "src/replicate_table5_table8.py",
]
raise SystemExit(subprocess.call(cmd))
