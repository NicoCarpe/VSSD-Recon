import argparse
import subprocess
import sys
import tempfile
import yaml

DEFAULT_CONFIG = "./configs/inference/vssd-recon/cmr25-cardiac.yaml"
CALLBACK_CLASS = "__main__.CustomWriter"

def main():
    p = argparse.ArgumentParser(description="Simple wrapper: --input/--output only")
    p.add_argument("--input", required=True, help="Path to input folder")
    p.add_argument("--output", required=True, help="Path to output folder")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="Inference LightningCLI YAML")
    args = p.parse_args()

    # Build a minimal override config so we don’t fight the CLI syntax
    override = {
        "data": {"init_args": {"data_path": args.input}},
        "trainer": {
            "callbacks": [
                {
                    "class_path": CALLBACK_CLASS,
                    "init_args": {
                        "output_dir": args.output,
                        "write_interval": "batch_and_epoch",
                    },
                }
            ]
        },
    }

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        yaml.safe_dump(override, tf)
        override_path = tf.name

    cmd = [
        sys.executable, "./main.py", "predict",
        "--config", args.config,
        "--config", override_path
    ]

    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
