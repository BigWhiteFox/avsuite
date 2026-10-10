#!/usr/bin/env bash
# Resumable downloader, tolerant to FlClash node switching / stalls.
# Tries FlClash proxy first, falls back to direct on each attempt.
# usage: dl.sh <url> <outfile>
url="$1"; out="$2"
mkdir -p "$(dirname "$out")"
proxy="http://127.0.0.1:7890"
for i in $(seq 1 3000); do
  # via proxy (abort if <8KB/s for 40s -> node switch/stall, then resume)
  curl -sL -C - --fail -x "$proxy" \
    --connect-timeout 20 --speed-time 40 --speed-limit 8192 \
    -o "$out" "$url" && { echo "DONE $out"; exit 0; }
  # direct fallback (hf-mirror/China CDN sources)
  curl -sL -C - --fail \
    --connect-timeout 20 --speed-time 40 --speed-limit 8192 \
    -o "$out" "$url" && { echo "DONE $out"; exit 0; }
  echo "[$(date +%H:%M:%S)] attempt $i rc=$? -> resume $(basename "$out") (have $(du -m "$out" 2>/dev/null | cut -f1)MB)"
  sleep 3
done
echo "FAILED $url"; exit 1
