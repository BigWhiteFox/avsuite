"""三方逐条对照探针:cascade vs v4 vs trad 的 AECMOS 逐条分布(DT 段重点)。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
SRC = H.parents[1] / "exp" / "07_nn_aec" / "outputs"
files = {
    "trad_wp06": SRC / "aecmos_trad_wp06.json",
    "nn_v4": SRC / "aecmos_nn_v4.json",
    "cascade_p1_1": H / "results" / "aecmos_cascade.json",
}
dfs = {}
for tag, p in files.items():
    rows = json.loads(p.read_text(encoding="utf-8"))["rows"]
    dfs[tag] = pd.DataFrame(rows)

m = dfs["trad_wp06"][["name", "echo_mos", "deg_mos"]].rename(
    columns={"echo_mos": "e_trad", "deg_mos": "d_trad"})
m = m.merge(dfs["nn_v4"][["name", "echo_mos", "deg_mos"]].rename(
    columns={"echo_mos": "e_v4", "deg_mos": "d_v4"}), on="name")
m = m.merge(dfs["cascade_p1_1"][["name", "echo_mos", "deg_mos"]].rename(
    columns={"echo_mos": "e_cas", "deg_mos": "d_cas"}), on="name")
sc = dfs["trad_wp06"].set_index("name")["scenario"]
m["scenario"] = m["name"].map(sc)

for s in ("dt", "fest", "nst"):
    sub = m[m.scenario == s]
    print(f"== {s}  n={len(sub)} ==")
    print(f"  echo: trad {sub.e_trad.mean():.3f}  v4 {sub.e_v4.mean():.3f}  "
          f"cascade {sub.e_cas.mean():.3f}")
    print(f"  deg : trad {sub.d_trad.mean():.3f}  v4 {sub.d_v4.mean():.3f}  "
          f"cascade {sub.d_cas.mean():.3f}")
    if s == "dt":
        print(sub[["name", "d_trad", "d_v4", "d_cas", "e_trad", "e_v4", "e_cas"]]
              .round(3).to_string(index=False))

print("== OVRL(60 条逐条均值,与 ZEROTRAIN 同口径)==")
print(f"  echo: trad {m.e_trad.mean():.3f}  v4 {m.e_v4.mean():.3f}  "
      f"cascade {m.e_cas.mean():.3f}")
print(f"  deg : trad {m.d_trad.mean():.3f}  v4 {m.d_v4.mean():.3f}  "
      f"cascade {m.d_cas.mean():.3f}")
