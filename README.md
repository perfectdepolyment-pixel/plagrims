# OriginCheck API

FastAPI + ONNX MiniLM (fastembed) + SQLite. CPU only, no PyTorch.

## Storage (SQLite)
One file: `$DATA_DIR/origincheck.db` (default `/data/origincheck.db`, override with `DB_PATH`).

| Table | Holds |
|---|---|
| `documents` | knowledge-base document metadata |
| `chunks` | text chunks + 384-dim float32 vectors (BLOB), cascade-deleted with their document |
| `reports` | saved similarity reports (full JSON) |

At startup all vectors are loaded into one NumPy matrix, so searching stays fast.
SQLite makes the data durable and transactional. WAL mode is enabled.
If the database file cannot be opened, the app falls back to an in-memory DB and logs a warning.

## Run locally
    docker build -t origincheck-api .
    docker run -p 8000:8000 -e API_KEY=secret -v $(pwd)/data:/data origincheck-api
    # docs: http://localhost:8000/docs

## Deploy on Render
1. Push this folder to GitHub.
2. Render -> New -> Blueprint (uses render.yaml) or New -> Web Service -> Runtime: Docker.
3. Set env vars: API_KEY, ALLOWED_ORIGINS (your frontend URL).
4. **Attach a persistent disk mounted at /data** (paid plan). Without it, the SQLite file
   is erased on every redeploy or restart.
5. Health check path: /health

## Endpoints
| Method | Route | Purpose |
|---|---|---|
| GET | /health | health check |
| POST | /v1/kb/documents | add file/text to the knowledge base |
| GET | /v1/kb/documents | list documents |
| DELETE | /v1/kb/documents/{id} | remove a document |
| POST | /v1/check | check a submission (saves a report unless save=false) |
| GET | /v1/reports | list saved reports |
| GET | /v1/reports/{id} | fetch one report |
| DELETE | /v1/reports/{id} | delete a report |

## Examples
    curl -X POST $URL/v1/kb/documents -H "X-API-Key: $KEY" \
      -F title="Past Project 2022" -F author="A. Student" -F year=2022 \
      -F doc_type="Internal repository" -F file=@project.pdf

    curl -X POST $URL/v1/check -H "X-API-Key: $KEY" -F file=@seminar.docx -F title="Seminar 1"
    curl $URL/v1/reports/<report_id> -H "X-API-Key: $KEY"

## Privacy note
Saved reports contain the full submission text. Use `-F save=false` to skip storing,
and delete reports with DELETE /v1/reports/{id} according to your retention policy.
