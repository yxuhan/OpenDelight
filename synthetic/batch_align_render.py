import os
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--render_root', type=str, default="workspace/20view")
parser.add_argument('--scan_root', type=str, default="data/ava256_4k")
opt, _ = parser.parse_known_args()


render_root = Path(opt.render_root)
align_render_script = "align_render.py"

if not render_root.is_dir():
    raise NotADirectoryError(f"render_root does not exist or is not a directory: {render_root}")

dname_list = sorted(p.name for p in render_root.iterdir() if p.is_dir())
for pth in tqdm(dname_list):
    cur_render_root = render_root / pth
    cur_scan_root = Path(opt.scan_root) / pth
    print(f"Processing {cur_render_root}")

    subprocess.run(
        [
            sys.executable,
            str(align_render_script),
            "--data_root",
            str(cur_render_root),
            "--scan_root",
            str(cur_scan_root),
        ],
        check=True,
    )
