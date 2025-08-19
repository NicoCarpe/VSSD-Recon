import os
import sys
import argparse
import subprocess

if __name__ == "__main__":
    argv = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?', default='/input', help='input directory')
    parser.add_argument('--output', type=str, nargs='?', default='/output', help='output directory')
    parser.add_argument("--config", default="/app/configs/inference/VSSD-Recon/cmr25-cardiac.yaml")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    cmd = [
        sys.executable, "/app/main.py", "predict",
        "--config", args.config,
        f"--data.init_args.data_path={args.input}",
        f"--trainer.callbacks[0].init_args.output_dir={args.output}"
    ]

    subprocess.run(cmd, check=True)

