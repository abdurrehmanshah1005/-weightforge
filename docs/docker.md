# Docker on Windows

Start Docker Desktop in Linux containers mode, then open PowerShell in the
WeightForge repository:

```powershell
docker compose config --quiet
docker compose up --build -d --wait
```

Open http://127.0.0.1:8000/ for the dashboard and
http://127.0.0.1:8000/docs for API exploration. The first build downloads the
Python base image, uv, and CPU PyTorch dependencies. Allow several minutes and
enough disk space for the image and build cache. No GPU is required.

```powershell
docker compose logs -f
docker compose stop
docker compose start
docker compose down
```

`down` removes the container and network but retains the named results volume.
Do not add `-v` if you want to keep experiment artifacts. Live sessions reset on
restart; their completed files remain in `/data/runs`.

Copy artifacts out of the running container:

```powershell
docker compose cp weightforge:/data/runs ./exported-runs
```

The image runs as a non-root user, uses locked CPU dependencies, exposes one
single-worker service and includes a health check. Compose binds the port to
loopback only. Source code, local virtual environments, checkpoints, credentials
and existing runs are excluded from the build context. Frontend assets are
installed inside the Python package; no Node server runs in the container.

The health check confirms the HTTP process is ready. It does not prove PyTorch
inference works. Test actual inference after startup with:

```powershell
uv run --locked python scripts/container_smoke.py
```

That script creates a real two-proposal guided run, checks accept then rollback,
and downloads final results. CI builds the container and runs this check on Linux.

Rebuild after changes:

```powershell
docker compose up --build -d --wait
```

For a busy port, stop the earlier local uvicorn process or change the host side
of the Compose port mapping. If Docker reports that its engine/pipe is missing,
start Docker Desktop and wait for it to become ready before retrying.
