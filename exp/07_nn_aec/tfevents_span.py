"""最小 tfevents 读取器：抽出 wall_time / step，用真实训练时长算卡时（t14 收档用）。

为什么不用文件 mtime：本地副本的 mtime = **拉回时间**（tar 解包），不是训练结束时间。
tfevents 记录里带 wall_time（Event 字段 1，fixed64 double）与 step（字段 2，varint），
解析首末两条即可得到真实起止；不需要 tensorboard/tensorflow 依赖。

用法：python exp/07_nn_aec/tfevents_span.py <目录或文件> [...]
"""
from __future__ import annotations

import struct
import sys
import time
from pathlib import Path


def _read_varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = 0
    val = 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, i
        shift += 7


def parse_record(data: bytes) -> tuple[float | None, int | None]:
    """只解 Event 的前几个字段：1=wall_time(fixed64 double)、2=step(varint)。"""
    i, n = 0, len(data)
    wall = step = None
    while i < n:
        key, i = _read_varint(data, i)
        field, wire = key >> 3, key & 7
        if wire == 1:                                    # 64-bit
            v = struct.unpack_from("<d", data, i)[0]
            i += 8
            if field == 1:
                wall = v
        elif wire == 0:                                  # varint
            v, i = _read_varint(data, i)
            if field == 2:
                step = v
        elif wire == 2:                                  # length-delimited
            ln, i = _read_varint(data, i)
            i += ln
        elif wire == 5:
            i += 4
        else:
            break
    return wall, step


def span(path: Path) -> dict:
    buf = path.read_bytes()
    i, n = 0, len(buf)
    first = last = None
    steps = []
    while i + 12 <= n:
        ln = struct.unpack_from("<Q", buf, i)[0]
        i += 12                                # length + length CRC
        data = buf[i:i + ln]
        i += ln + 4                            # data + data CRC
        wall, step = parse_record(data)
        if wall:
            if first is None:
                first = wall
            last = wall
        if step is not None:
            steps.append(step)
    if first is None:
        return {}
    return {"file": path.name, "start": first, "end": last, "hours": (last - first) / 3600.0,
            "max_step": max(steps) if steps else None, "n_records": len(steps)}


def main() -> int:
    targets = []
    for a in sys.argv[1:]:
        p = Path(a)
        targets.extend(sorted(p.rglob("events.out.tfevents.*")) if p.is_dir() else [p])
    total = 0.0
    for t in targets:
        s = span(t)
        if not s:
            print(f"  {t}: 无有效记录")
            continue
        total += s["hours"]
        print(f"  {t.parent.parent.name if t.parent.name else t.parent.name:24s} "
              f"{time.strftime('%m-%d %H:%M', time.localtime(s['start']))} → "
              f"{time.strftime('%m-%d %H:%M', time.localtime(s['end']))} = "
              f"{s['hours']:6.2f} h  step≤{s['max_step']}")
    print(f"  ---- 简单相加 {total:.2f} h（并发 run 不可直接相加，见报告口径说明）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
