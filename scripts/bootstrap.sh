#!/usr/bin/env bash
set -euo pipefail
# Blocker 11.1: .env is gitignored in main checkout, missing in worktree. Symlink or copy it.
SRC="/home/azidozide/projects/syndicate_/.env"
DST="$(cd "$(dirname "$0")/.." && pwd)/.env"
if [ ! -f "$SRC" ]; then
  echo "ERROR: source .env not found at $SRC" >&2; exit 1
fi
if [ -f "$DST" ] || [ -L "$DST" ]; then
  echo ".env already exists at $DST"
else
  if ln -s "$SRC" "$DST" 2>/dev/null; then
    echo "Symlinked $SRC -> $DST"
  else
    cp "$SRC" "$DST"
    echo "Copied $SRC -> $DST (symlink failed, e.g. cross-fs)"
  fi
fi
# Verify API base URLs reachable (HEAD/GET, don't require auth)
set -a; source "$DST"; set +a
for var in OPENAI_BASE_URL TENSORMUX_BASE_URL; do
  url="${!var:-}"
  if [ -z "$url" ]; then echo "WARN: $var empty" >&2; continue; fi
  echo -n "Checking $var=$url ... "
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$url" || echo "curl-fail")
  echo "HTTP $code"
  if [[ "$code" == "curl-fail" || "$code" == "000" ]]; then
    echo "ERROR: $var unreachable" >&2; exit 1
  fi
done
echo "bootstrap ok"
