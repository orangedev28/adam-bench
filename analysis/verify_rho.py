"""Verify the proxy rho_l = eta_t * sqrt(C_l) / ||W_l|| against measured step updates and angles.

Reads result.json files produced by v4 (which contain rel_upd, ang_upd, rho_step) from E6, E6c,
E1c, E5, E5b, prints a summary table, and writes analysis/out/verify_rho.json.
"""
import collections
import glob
import json
import math
import os
import statistics as st

os.makedirs("analysis/out", exist_ok=True)


def imbalance(vals):
    pos = [v for v in vals if v > 0]
    return max(pos) / min(pos) if len(pos) >= 2 else float("nan")


def main():
    runs = collections.defaultdict(list)
    for f in sorted(glob.glob("results/followup/E*/*/runs/*_final_*/result.json")):
        d = json.load(open(f))
        h = d.get("history", [])
        if h and "rel_upd" in h[0] and "ang_upd" in h[0] and "rho_step" in h[0]:
            runs[(d["cell"], d["optimizer"])].append(d)

    if not runs:
        print("No runs with rel_upd / ang_upd / rho_step found under results/followup/.")
        return

    out = {}
    for (cell, opt), rs in sorted(runs.items()):
        epochs_total = rs[0]["epochs"]
        check_eps = [ep for ep in (10, 25, 40, 60, 80) if ep <= epochs_total]
        if not check_eps:
            check_eps = [epochs_total]
        n_layers = len(rs[0]["history"][0]["rel_upd"])
        per_ep = {}
        i_proxy_list, i_rel_list, i_ang_list = [], [], []
        rel_rho_all, ang_rho_all, ang_rel_all = [], [], []

        for ep in check_eps:
            hs = [r["history"][ep - 1] for r in rs if len(r["history"]) >= ep and "rel_upd" in r["history"][ep - 1]]
            if not hs:
                continue
            rho_l = [st.mean(h["rho_step"][i] for h in hs) for i in range(n_layers)]
            rel_l = [st.mean(h["rel_upd"][i] for h in hs) for i in range(n_layers)]
            ang_l = [st.mean(h["ang_upd"][i] for h in hs) for i in range(n_layers)]
            i_proxy = imbalance(rho_l)
            i_rel = imbalance(rel_l)
            i_ang = imbalance(ang_l)
            per_ep[str(ep)] = {
                "rho_step": rho_l,
                "rel_upd": rel_l,
                "ang_upd": ang_l,
                "rel_over_rho": [r / p if p > 0 else None for r, p in zip(rel_l, rho_l)],
                "ang_over_rho": [a / p if p > 0 else None for a, p in zip(ang_l, rho_l)],
                "I_proxy": i_proxy,
                "I_rel": i_rel,
                "I_ang": i_ang,
            }
            if ep in (10, 25, 40) or epochs_total != 50:
                i_proxy_list.append(i_proxy)
                i_rel_list.append(i_rel)
                i_ang_list.append(i_ang)
            rel_rho_all.extend(r / p for r, p in zip(rel_l, rho_l) if p > 0)
            ang_rho_all.extend(a / p for a, p in zip(ang_l, rho_l) if p > 0)
            ang_rel_all.extend(a / r for a, r in zip(ang_l, rel_l) if r > 0)

        rec = {
            "n_seeds": len(rs),
            "n_layers": n_layers,
            "I_proxy_mean": st.mean(i_proxy_list) if i_proxy_list else None,
            "I_rel_mean": st.mean(i_rel_list) if i_rel_list else None,
            "I_ang_mean": st.mean(i_ang_list) if i_ang_list else None,
            "rel_over_rho_mean": st.mean(rel_rho_all) if rel_rho_all else None,
            "ang_over_rho_mean": st.mean(ang_rho_all) if ang_rho_all else None,
            "ang_over_rel_mean": st.mean(ang_rel_all) if ang_rel_all else None,
            "epochs": per_ep,
        }
        out[f"{cell}|{opt}"] = rec
        print(
            f"{cell} {opt:13s} (n={len(rs)}): "
            f"I_proxy={rec['I_proxy_mean']:.2f}  I_rel={rec['I_rel_mean']:.2f}  I_ang={rec['I_ang_mean']:.2f} | "
            f"rel/rho={rec['rel_over_rho_mean']:.3f}  ang/rho={rec['ang_over_rho_mean']:.3f}  "
            f"ang/rel={rec['ang_over_rel_mean']:.3f}"
        )

    json.dump(out, open("analysis/out/verify_rho.json", "w"), indent=1)
    print("Wrote analysis/out/verify_rho.json")


if __name__ == "__main__":
    main()
