#!/usr/bin/env bash
# Download AEC-Challenge ICASSP2022 test set (3 scenarios) via media.githubusercontent
set -u
proxy="http://127.0.0.1:7890"
outbase="/d/data/aec-challenge/test_set_icassp2022"
mkdir -p "$outbase"
for s in doubletalk farend-singletalk nearend-singletalk; do
  mkdir -p "$outbase/$s"
  curl -s -x "$proxy" -m 60 "https://api.github.com/repos/microsoft/AEC-Challenge/contents/datasets/test_set_icassp2022/$s" \
   | python -c "import json,sys; [print(x['name']) for x in json.load(sys.stdin)]" | tr -d '\r' > "$outbase/$s/.filelist" || { echo "LIST FAIL $s"; exit 1; }
  n=$(wc -l < "$outbase/$s/.filelist")
  echo "[$(date +%H:%M:%S)] $s: $n files"
  i=0
  while read -r fn; do
    i=$((i+1))
    if [ -s "$outbase/$s/$fn" ] && [ "$(stat -c%s "$outbase/$s/$fn" 2>/dev/null || echo 0)" -gt 1000 ]; then continue; fi
    for try in $(seq 1 12); do
      curl -sL -x "$proxy" --fail --connect-timeout 20 --speed-time 30 --speed-limit 4096 \
        -o "$outbase/$s/$fn" \
        "https://media.githubusercontent.com/media/microsoft/AEC-Challenge/main/datasets/test_set_icassp2022/$s/$fn" && break
      sleep 2
    done
    if [ $((i % 100)) -eq 0 ]; then echo "[$(date +%H:%M:%S)] $s: $i/$n"; fi
  done < "$outbase/$s/.filelist"
  echo "[$(date +%H:%M:%S)] scenario $s DONE"
done
echo "ALL AEC TESTSET DONE"
