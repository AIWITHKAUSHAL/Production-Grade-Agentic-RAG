#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found. Copy .env.example to .env and fill in your API keys."
    exit 1
fi

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt -q

uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
