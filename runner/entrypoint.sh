#!/bin/bash
set -euo pipefail

# Falls uv ein .venv erzeugt hat (Standard), aktivieren:
if [ -f "/app/.venv/bin/activate" ]; then
    # Temporär -u deaktivieren, um 'source' korrekt zu laden
    set +u
    . /app/.venv/bin/activate
    set -u
fi

# Starten des Python Scripts
exec python /app/kafka_consumer.py