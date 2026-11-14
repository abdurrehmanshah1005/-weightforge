# Two-service local deployment

```powershell
docker compose up --build -d --wait
docker compose logs --tail 100
docker compose down
```

Open http://127.0.0.1:8000/ and http://127.0.0.1:8000/api/docs.
Frontend: unprivileged Nginx on internal port 8080. Backend: one FastAPI worker on
internal port 8000, no published host port. SSE proxy buffering is disabled.
Both containers drop capabilities and disallow gaining privileges. Health checks
gate frontend startup; restart policies support ordinary process restarts.

The `weightforge-runs` volume stores results under `/data/runs` and imported models
under `/data/runs/models`. Existing result folders from the previous deployment
remain discoverable. `docker compose down` retains data; do not add `--volumes`
unless you intend to erase it. Back up this volume before upgrades.

For MNIST, first run the README preparation command on the host. `./runs/data` is
mounted read-only at `/data/datasets`. No dataset downloads happen during inference.
The checkpoint is imported through the model library. Local-development runs and
Docker-volume runs are separate stores.

Compose reads optional credentials from `.env`. After changing them:

```powershell
docker compose up -d --force-recreate backend
```

Do not publish the backend or change the host binding to all interfaces for public
deployment. This profile is for one trusted local user; it does not supply accounts,
TLS, hostile-upload process isolation, resource quotas or durable task recovery.
`weights_only=True` restricts checkpoint loading but is not a resource sandbox.

CI builds both services, loads the actual frontend, then runs
`python scripts/container_smoke.py` against the `/api` proxy. This checks a real
scripted XOR accept/rollback sequence and exported results. Optional paid LLM calls
are never made in CI. See `verification.md` for local execution limitations.
