# CMRxRecon2025 Docker submission package

> **This directory contains no tests.** The name comes from the challenge's own
> submission template ("test phase"). It is a self-contained, inference-only copy of the
> project that gets baked into the Docker image the organisers run. Consider renaming it
> `docker/` or `submission/`.

The challenge evaluates entries by running a container against held-out data it mounts
at `/input`, expecting reconstructions at `/output`. The container has no network access
and no build toolchain, so the package here is deliberately narrower than the root tree.

## How it runs

```text
dockerfile              COPY test /app; pip install -r /app/configs/requirements.txt
docker-entrypoint.sh    cd /output → before_run.py → inference.py → after_run.py
test/inference.py       --input/--output → writes an override YAML → main.py predict
test/main.py            same LightningCLI entry point as the root
```

`inference.py` exists because the organisers' contract is `--input DIR --output DIR`,
while `main.py` is a LightningCLI. It synthesises a small override config that redirects
`data.init_args.data_path` and `CustomWriter.output_dir`, then appends it after the base
inference config — later `--config` files win, so nothing else needs editing.

Build and smoke-test locally from the repository root:

```bash
bash docker-test.sh          # docker build -f dockerfile . && docker run --gpus all ...
```

The checkpoint is **not** committed. `test/configs/inference/vssd-recon/cmr25-cardiac.yaml`
expects it at `./checkpoints/last.ckpt` relative to `/app`; mount or `COPY` it in before
building a submission image.

`../before_run.py` and `../after_run.py` are the organisers' unmodified hooks. `after_run`
sends a completion email through a hardcoded third-party SMTP account and is a no-op
unless `SMTP_PASSWORD` is set, so it exits cleanly in normal use.
