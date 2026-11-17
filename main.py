"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Method (standard library only, offline, deterministic):
  The case file itself carries 38-42 past big launches ("comparables")
  with their full release-day hourly Twitch curves.  Hour 1's print
  anchors the scale, but the *shape* of the rest of the day depends
  strongly on how big hour 1 already is: launches whose hour 1 was
  already huge (a simultaneous global unlock) barely grow, while
  launches whose hour 1 was small (only some regions unlocked yet)
  grow several-fold once the US unlocks.  Across the comparables,
  log(hour-1 viewers) and log(peak / hour-1) correlate about -0.94.

  So for each hour h = 2..24 we forecast log(viewers_h / viewers_1)
  two ways, then blend them in log space (50/50):
  1. A per-hour regression of log-ratio on log(hour-1 viewers),
     slope clipped to [-1, 0], with the empirical quantiles of its
     residuals as the spread.  This extrapolates sensibly to a very
     small or very large hour 1.
  2. A k-nearest-neighbour forecast (K = 8) in log(hour-1) space:
     the empirical quantiles of the log-ratios of the 8 launches
     whose hour 1 was closest in size to this case's hour 1.

  The blend scored about 0.214 mean pinball loss on the four
  practice launches (Marvel's Wolverine, Red Dead Redemption 2,
  God of War Ragnarok, Marvel's Spider-Man 2), against 0.387 for
  the house baseline (flat at hour 1 with a widening spread), and
  about 0.41 in leave-one-out scoring over the comparables
  themselves, against 0.93 for the baseline.

  Output: {"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}
  for h = 2..24, quantiles non-decreasing, in (0, 1e9].
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
KNN_K = 8


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """Round to 2 decimals, half up, as the starter does."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def quantiles(sorted_vals, probs):
    """Linear-interpolated quantiles of an already sorted list."""
    n = len(sorted_vals)
    out = []
    for p in probs:
        if n == 1:
            out.append(sorted_vals[0])
            continue
        pos = p * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        out.append(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)
    return out


def release_day_map(comp):
    """Hour (0..23) -> viewers for a comparable's release day."""
    try:
        day = parse_time(comp["day_start"])
    except Exception:
        return {}
    hours = {}
    for row in comp.get("hourly") or []:
        try:
            when = parse_time(row["from"])
            viewers = float(row["viewers"])
        except Exception:
            continue
        delta = int((when - day).total_seconds() // 3600)
        if 0 <= delta < 24 and viewers > 0:
            hours[delta] = viewers
    return hours


def baseline_quantiles(p1, hour):
    """House baseline: flat at hour 1, spread widening with sqrt(h-1)."""
    return [p1 * math.exp(z * 0.15 * math.sqrt(hour - 1)) for z in Z]


def regression_quantiles(maps, p1, hour):
    """Per-hour log-ratio regression on log(hour-1), empirical spread."""
    idx = hour - 1
    xs = []
    ys = []
    for m in maps:
        h1 = m.get(0)
        val = m.get(idx)
        if h1 and h1 > 1000 and val:
            xs.append(math.log(h1))
            ys.append(math.log(val / h1))
    if len(xs) < 5:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    slope = 0.0
    if vx > 0:
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx
    slope = max(-1.0, min(0.0, slope))
    intercept = my - slope * mx
    pred = intercept + slope * math.log(max(p1, 1.0))
    resid = sorted(y - (intercept + slope * x) for x, y in zip(xs, ys))
    spread = quantiles(resid, LEVELS)
    return [p1 * math.exp(pred + r) for r in spread]


def knn_quantiles(named_maps, p1, hour):
    """Empirical quantiles from the K launches with closest hour 1."""
    idx = hour - 1
    log_p1 = math.log(max(p1, 1.0))
    ranked = []
    for _name, m in named_maps:
        h1 = m.get(0)
        if h1 and h1 > 0:
            ranked.append((abs(math.log(h1) - log_p1), m))
    ranked.sort(key=lambda item: item[0])
    logs = []
    for _dist, m in ranked[:KNN_K]:
        val = m.get(idx)
        if val:
            logs.append(math.log(val / m[0]))
    if len(logs) < 3:
        return None
    logs.sort()
    qs = quantiles(logs, LEVELS)
    return [p1 * math.exp(q) for q in qs]


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("unexpected levels")
    known = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if known is None:
        raise ValueError("no hour-1 print")
    p1 = max(1.0, float(known["viewers"]))

    named_maps = []
    for comp in case.get("comparables") or []:
        m = release_day_map(comp)
        # A usable launch curve: it was already live at hour 1 and
        # has most of the day recorded.  The GTA VI target-history
        # comparable has no hour 1 on its "day", so it drops out here.
        if 0 in m and len(m) >= 20:
            named_maps.append((comp.get("name", ""), m))
    maps = [m for _name, m in named_maps]

    hours_out = []
    for row in case.get("forecast") or []:
        hour = int(row["hour"])
        reg = regression_quantiles(maps, p1, hour)
        knn = knn_quantiles(named_maps, p1, hour)
        if reg is not None and knn is not None:
            qs = [math.exp(0.5 * math.log(reg[i]) + 0.5 * math.log(knn[i]))
                  for i in range(5)]
        elif reg is not None:
            qs = reg
        elif knn is not None:
            qs = knn
        else:
            qs = baseline_quantiles(p1, hour)
        # Enforce the output contract: finite, in (0, 1e9], sorted.
        clean = []
        prev = 0.0
        for q in qs:
            v = q
            if not math.isfinite(v) or v <= 0:
                v = max(prev, 1.0)
            v = min(v, 1_000_000_000.0)
            v = max(v, prev)
            clean.append(round2(v))
            prev = clean[-1]
        hours_out.append({"hour": hour, "q": clean})
    # Sort by hour so two runs write the same bytes whatever order
    # the case listed its forecast rows in.
    hours_out.sort(key=lambda row: row["hour"])
    return {"hours": hours_out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
