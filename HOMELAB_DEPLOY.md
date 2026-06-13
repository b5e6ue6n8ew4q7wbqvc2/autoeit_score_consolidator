# Homelab Deployment Notes: autoeit-score-consolidator

## What This App Is

A stateless, single-page Streamlit web app for consolidating AutoEIT score exports. Users drag and drop (or click to upload) a zip file through the browser; the app detects the zip type automatically, processes everything in memory, and produces a download. There is no database, no persistent storage, no authentication, and no external network calls.

### Supported ZIP types (auto-detected from contents)

| Type | Contents | Output |
|------|----------|--------|
| AUDIO_CSV | bio.csv + submissions.csv + MP3s | Single consolidated zip (CSVs + filtered MP3s) |
| CSV-only | bio.csv + submissions.csv | Consolidated CSV zip; optional separate audio zip if one is also uploaded |
| Audio-only | MP3s only | Requires a second CSV zip upload; produces separate consolidated audio zip |

AUDIO_CSV is the current platform format. CSV-only and Audio-only are supported for backward compatibility.

## Repository

Source is on the host machine at the path this file was found in. The deploy directory on homelab is `/home/eve/autoeit-score-consolidator/`. There is also a GitHub remote:
`https://github.com/b5e6ue6n8ew4q7wbqvc2/autoeit_score_consolidator.git`

## Deployment

All required files are in the project directory:

```
Dockerfile
docker-compose.yml
.dockerignore
.streamlit/config.toml
app.py
requirements.txt
```

### Build and run

```bash
docker compose up -d --build
```

### Stop

```bash
docker compose down
```

### View logs

```bash
docker compose logs -f autoeit-score-consolidator
```

## Port

| Host port | Container port | Protocol |
|-----------|---------------|----------|
| 8501      | 8501          | TCP (HTTP) |

The app is served at `http://<host-ip>:8501`. If 8501 conflicts with another service, change the left side of the port mapping in `docker-compose.yml` (e.g., `"8502:8501"`).

## Health Check

The container has a built-in Docker health check:

```
GET http://localhost:8501/_stcore/health
interval: 30s | timeout: 10s | start-period: 15s | retries: 3
```

## Resource Profile

- **CPU/RAM:** Very light. Pure Python data processing on small CSV files. No GPU, no ML models.
- **Disk:** Image is ~200-300 MB (python:3.12-slim + streamlit + pandas).
- **Network:** Inbound only on port 8501. No outbound calls at runtime.
- **Storage volumes:** None. No data is written to disk by the container.

## Restart Policy

Set to `unless-stopped` — the container will restart automatically after host reboots or crashes unless explicitly stopped with `docker compose down`.

## Dependencies (pip)

- `streamlit` (unpinned — latest at build time)
- `pandas` (unpinned — latest at build time)

If a future build breaks due to a dependency update, pin the versions in `requirements.txt` by checking the working versions with `docker exec autoeit-score-consolidator pip freeze`.

## Notes

- No `.env` file or secrets are needed.
- No volumes need to be mounted.
- The `example_files/` directory in the repo is sample data for development only and is excluded from the image via `.dockerignore`.
- If deploying behind a reverse proxy (e.g., Nginx, Traefik, Caddy), note that Streamlit uses WebSockets. Ensure the proxy is configured to upgrade WebSocket connections, otherwise the UI will not function correctly.
