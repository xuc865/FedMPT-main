#!/usr/bin/env bash
# Backward-compatible wrapper; prefer run.sh
exec bash "$(dirname "$0")/run.sh" "$@"
