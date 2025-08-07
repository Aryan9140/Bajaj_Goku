##!/usr/bin/env bash
# start.sh

# Run FastAPI with uvicorn (recommended for production)
# --host 0.0.0.0: allows external access
# --port 10000: Render default port
# --workers: (optional, set >1 for concurrency, but comment out for async/await style apps)

uvicorn run:app --host 0.0.0.0 --port 10000
