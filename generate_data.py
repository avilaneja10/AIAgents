"""
Generate the CS839 HW2 multi-agent dataset and its ground-truth answer key.

Agent-visible output (data/):
  airlines/<CC>.csv   one flights file per carrier (DL, AA, UA, WN, B6, AS)
  weather.csv         hourly METAR-style weather per airport (local time)
  airports.csv        airport metadata incl. UTC offset
  README.md           data dictionary + conventions (given to every system)
  weather_codes.md    weather-code legend + the "bad weather" rule

Hidden output (NOT in data/, never shown to agents):
  tasks.json          the 8 task prompts, categories, expected answers, and the
                      answer each trap would produce (for diagnosing failures)

Deliberate traps (all detectable from the data + README):
  1. AS logs date/times in UTC (columns *_utc) -> must convert to local via airports.csv
  2. B6 file contains exact duplicate rows      -> flight_id is unique per flight
  3. Flights that depart after midnight         -> actual_dep "00:20" vs sched "23:50"
                                                   means +30 min, not -1410 min
  4. Weather codes: only some non-clear codes count as "bad weather"

The answer key is computed TWICE: from the internal ground truth, and by a
reference solver that reads the written files (with every trap handled). The
two must agree, which checks that every task is solvable from the files alone.

Run:  python generate_data.py              (default seed 853)
      python generate_data.py --seed N --search   (fresh dataset for stability reruns)
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

DELAY_THRESHOLD = 15            # delayed := departure delay > 15 min
START, END = date(2025, 1, 1), date(2025, 2, 28)
LOOKUP_DAY = date(2025, 2, 14)  # T7

AIRPORTS = {  # code: (city, IANA tz, winter UTC offset in hours; no DST in Jan/Feb)
    "ATL": ("Atlanta", "America/New_York", -5),
    "JFK": ("New York", "America/New_York", -5),
    "ORD": ("Chicago", "America/Chicago", -6),
    "DFW": ("Dallas-Fort Worth", "America/Chicago", -6),
    "DEN": ("Denver", "America/Denver", -7),
    "LAX": ("Los Angeles", "America/Los_Angeles", -8),
    "SEA": ("Seattle", "America/Los_Angeles", -8),
}

# carrier: (flights/day, base probability of a >15-min delay)
CARRIERS = {
# Big carriers late more often, small carriers less, so averaging per-carrier rates
# (instead of pooling counts) visibly biases the network rate (T5 trap).
    "WN": (9, 0.30),
    "DL": (7, 0.20),
    "AA": (6, 0.20),
    "UA": (5, 0.18),
    "B6": (3, 0.08),
    "AS": (3, 0.05),
}
UTC_CARRIER = "AS"
DUP_CARRIER, N_DUPS = "B6", 12

# Route weights per carrier (undirected pairs). Chosen so that the global top-5
# routes are NOT simply the union of each carrier's top routes (T6 trap):
# ATL-DFW is mid-ranked for three carriers but top-3 overall.
ROUTES = {
    "WN": {"DEN-LAX": 3, "DFW-LAX": 2.5, "DEN-SEA": 2, "ATL-DFW": 1.5, "DEN-ORD": 1.5, "DEN-DFW": 1.2, "DFW-ORD": 1},
    "DL": {"ATL-JFK": 3, "ATL-LAX": 2.5, "ATL-ORD": 2, "ATL-SEA": 1.5, "ATL-DFW": 1.4, "ATL-DEN": 1},
    "AA": {"DFW-ORD": 2.5, "DFW-LAX": 2.5, "DFW-JFK": 2, "DEN-DFW": 1.4, "ATL-DFW": 1.3, "DFW-SEA": 0.8},
    "UA": {"LAX-ORD": 2.5, "JFK-ORD": 2, "DEN-SEA": 2, "DEN-LAX": 1.5, "DEN-DFW": 1.3, "ATL-ORD": 1},
    "B6": {"JFK-LAX": 3, "ATL-JFK": 1.5, "JFK-SEA": 1.5, "JFK-ORD": 1},
    "AS": {"LAX-SEA": 3, "JFK-SEA": 1, "DEN-SEA": 1},
}

# Weather: per airport-day, probability of an event of each code. "mild" codes are
# look-alikes that are NOT bad weather (so "anything not CLR" over-counts).
WX_PROFILE = {
    "ATL": {"mild": {"-RA": .15, "RA": .10, "BR": .10}, "bad": {"TS": .06, "+RA": .08, "FG": .06, "SN": .01}},
    "JFK": {"mild": {"-RA": .12, "RA": .08, "BR": .10}, "bad": {"SN": .10, "+RA": .06, "FG": .06, "TS": .01}},
    "ORD": {"mild": {"-RA": .08, "BR": .10, "RA": .05}, "bad": {"SN": .18, "+RA": .03, "FG": .05, "TS": .01}},
    "DFW": {"mild": {"-RA": .10, "RA": .08, "BR": .08}, "bad": {"TS": .07, "+RA": .06, "FG": .06, "SN": .02}},
    "DEN": {"mild": {"BR": .06, "-RA": .03}, "bad": {"SN": .18, "FG": .04}},
    "LAX": {"mild": {"BR": .15, "-RA": .08}, "bad": {"FG": .10, "+RA": .03}},
    "SEA": {"mild": {"RA": .25, "-RA": .25, "BR": .15}, "bad": {"+RA": .08, "FG": .08, "SN": .03}},
}
WX_SCALE = {"mild": 2.0, "bad": 3.0}   # event-rate multipliers (tune how often weather matters)
BAD_CODES = {"TS", "SN", "+RA"}
FOG_VIS_LIMIT = 1.0


def is_bad(code, vis):
    return code in BAD_CODES or (code == "FG" and vis < FOG_VIS_LIMIT)


def hhmm(minutes):
    minutes %= 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def days():
    d = START
    while d <= END:
        yield d
        d += timedelta(days=1)


# ---------------------------------------------------------------- generation
def visibility(rng, code):
    lo, hi = {"CLR": (10, 10), "-RA": (6, 10), "RA": (3, 7), "+RA": (1, 3), "BR": (1, 6),
              "FG": (0.25, 2.5), "SN": (0.5, 3), "TS": (2, 6)}[code]
    return round(rng.uniform(lo, hi) * 4) / 4  # quarter-mile steps, like METAR


def gen_weather(rng):
    wx = {}
    for ap, prof in WX_PROFILE.items():
        for d in days():
            hours = [("CLR", 10.0)] * 24
            for group, (lo, hi) in (("mild", (3, 10)), ("bad", (3, 10))):
                for code, p in prof[group].items():
                    if rng.random() < min(1.0, p * WX_SCALE[group]):
                        start, dur = rng.randrange(24), rng.randint(lo, hi)
                        for h in range(start, min(24, start + dur)):
                            hours[h] = (code, visibility(rng, code))
            if ap == "JFK" and d == LOOKUP_DAY:          # Valentine's Day snowstorm (T7)
                for h in range(5, 22):
                    hours[h] = ("SN", visibility(rng, "SN"))
            for h, (code, vis) in enumerate(hours):
                wx[(ap, d, h)] = (code, vis)
    return wx


def gen_flights(rng, wx):
    flights = []
    for cc, (per_day, p_base) in CARRIERS.items():
        pairs, weights = zip(*ROUTES[cc].items())
        n = 0
        for d in days():
            for _ in range(per_day + rng.choice([-1, 0, 0, 1])):
                a, b = rng.choices(pairs, weights)[0].split("-")
                origin, dest = (a, b) if rng.random() < 0.5 else (b, a)
                sched = rng.randrange(300, 1440, 5)       # 05:00 .. 23:55 local
                code, vis = wx[(origin, d, sched // 60)]
                bad = is_bad(code, vis)
                p_cancel = 0.01 + (0.10 if bad else 0.0)
                if cc == "B6" and d == LOOKUP_DAY and "JFK" in (origin, dest):
                    p_cancel = 0.6
                cancelled = rng.random() < p_cancel
                delay = None
                if not cancelled:
                    if rng.random() < p_base + (0.35 if bad else 0.0):
                        delay = DELAY_THRESHOLD + 1 + min(int(rng.expovariate(1 / 40)), 300)
                    else:
                        delay = rng.randint(-10, DELAY_THRESHOLD)
                n += 1
                flights.append(dict(carrier=cc, flight_id=f"{cc}{1000 + n}", date=d, origin=origin,
                                    dest=dest, sched=sched, delay=delay, cancelled=cancelled, bad=bad))
    return flights


README_MD = """# Flight operations dataset (January-February 2025)

## Files
| file | contents |
|---|---|
| `airlines/<CARRIER>.csv` | one file per carrier: DL (Delta), AA (American), UA (United), WN (Southwest), B6 (JetBlue), AS (Alaska) |
| `weather.csv` | hourly weather observation at each airport |
| `airports.csv` | airport metadata: `airport, city, timezone, utc_offset_hours` |
| `weather_codes.md` | meaning of weather codes and the definition of **bad weather** |

### Flights columns
`flight_id, date, origin, dest, sched_dep, actual_dep, cancelled`
- `flight_id` is unique: every flight appears exactly once. The export may contain repeated records.
- `date` is the scheduled departure date; `sched_dep` / `actual_dep` are clock times `HH:MM`.
- A delayed flight may depart after midnight, i.e. on the day after `date`.
- `actual_dep` is empty if the flight was cancelled. `cancelled` is `true` / `false`.
- Times are local time at the origin airport, **unless the column name ends in `_utc`**.

### Weather columns
`airport, date, hour, wx_code, visibility_mi` -- local time at that airport; `hour` h covers h:00-h:59.

## Conventions
- Departure delay (minutes) = actual departure - scheduled departure. Early departures are negative.
- A flight is **delayed** if its departure delay is **more than 15 minutes**.
- **Delay rate** = delayed flights / completed (non-cancelled) flights. Cancelled flights are excluded
  from both numerator and denominator. Report rates as percentages with 1 decimal.
- Dates, months and hours refer to local time at the origin airport.
"""

WEATHER_CODES_MD = """# Weather codes (METAR style)

| code | meaning |
|---|---|
| `CLR` | clear |
| `-RA` | light rain |
| `RA` | moderate rain |
| `+RA` | heavy rain |
| `BR` | mist |
| `FG` | fog |
| `SN` | snow |
| `TS` | thunderstorm |

`visibility_mi` is visibility in statute miles.

## Bad weather
An airport-hour counts as **bad weather** if and only if:
- `wx_code` is `TS`, `SN` or `+RA`; **or**
- `wx_code` is `FG` **and** `visibility_mi` < 1.0.

Everything else (`CLR`, `-RA`, `RA`, `BR`, and `FG` with visibility >= 1.0) is **not** bad weather.
"""


# ------------------------------------------------------------------- writing
def write_data(out, flights, wx, rng):
    (out / "airlines").mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(README_MD)
    (out / "weather_codes.md").write_text(WEATHER_CODES_MD)
    pd.DataFrame([dict(airport=k, city=v[0], timezone=v[1], utc_offset_hours=v[2]) for k, v in AIRPORTS.items()]
                 ).to_csv(out / "airports.csv", index=False)
    pd.DataFrame([dict(airport=ap, date=d.isoformat(), hour=h, wx_code=c, visibility_mi=v)
                  for (ap, d, h), (c, v) in sorted(wx.items())]).to_csv(out / "weather.csv", index=False)

    for cc in CARRIERS:
        rows = []
        for f in (f for f in flights if f["carrier"] == cc):
            local_s = datetime.combine(f["date"], datetime.min.time()) + timedelta(minutes=f["sched"])
            local_a = None if f["cancelled"] else local_s + timedelta(minutes=f["delay"])
            if cc == UTC_CARRIER:
                shift = timedelta(hours=-AIRPORTS[f["origin"]][2])
                us = local_s + shift
                rows.append(dict(flight_id=f["flight_id"], date_utc=us.date().isoformat(), origin=f["origin"],
                                 dest=f["dest"], sched_dep_utc=us.strftime("%H:%M"),
                                 actual_dep_utc="" if local_a is None else (local_a + shift).strftime("%H:%M"),
                                 cancelled=str(f["cancelled"]).lower()))
            else:
                rows.append(dict(flight_id=f["flight_id"], date=f["date"].isoformat(), origin=f["origin"],
                                 dest=f["dest"], sched_dep=hhmm(f["sched"]),
                                 actual_dep="" if local_a is None else local_a.strftime("%H:%M"),
                                 cancelled=str(f["cancelled"]).lower()))
        if cc == DUP_CARRIER:  # exact duplicate rows, never on the T7 lookup day
            pool = [r for r in rows if r["date"] != LOOKUP_DAY.isoformat()]
            rows += [dict(r) for r in rng.sample(pool, N_DUPS)]
        df = pd.DataFrame(rows)
        dcol, scol = ("date_utc", "sched_dep_utc") if cc == UTC_CARRIER else ("date", "sched_dep")
        df.sort_values([dcol, scol, "flight_id"]).to_csv(out / "airlines" / f"{cc}.csv", index=False)


# ---------------------------------------------------- tasks + answer computing
def pair(o, d):
    return "-".join(sorted((o, d)))


def rate(sub):
    nc = sub[~sub.cancelled]
    return round(100 * (nc.delay > DELAY_THRESHOLD).mean(), 1)


def answers(df):
    """All 8 answers from a normalized frame with columns:
    carrier, flight_id, date (datetime.date, local), origin, dest, sched (min), delay, cancelled, bad."""
    df = df.copy()
    df["route"] = [pair(o, d) for o, d in zip(df.origin, df.dest)]
    df["month"] = [d.month for d in df.date]
    both = df
    out = {}

    out["T1"] = {cc: rate(df[(df.carrier == cc) & (df.month == 2)]) for cc in CARRIERS}

    t2 = {}
    for cc in CARRIERS:
        sub = both[both.carrier == cc]
        counts = sub.route.value_counts()
        nc = sub[~sub.cancelled]
        g = nc.groupby("route").delay.agg(n="size", rate=lambda s: round(100 * (s > DELAY_THRESHOLD).mean(), 1))
        g = g[g.n >= 20].sort_values("rate", ascending=False)
        t2[cc] = {"busiest": counts.index[0], "busiest_n": int(counts.iloc[0]),
                  "busiest_tie": bool(len(counts) > 1 and counts.iloc[0] == counts.iloc[1]),
                  "worst": g.index[0], "worst_rate": float(g.rate.iloc[0]),
                  "worst_tie": bool(len(g) > 1 and g.rate.iloc[0] == g.rate.iloc[1])}
    out["T2"] = t2

    sub = both[both.carrier.isin(["DL", "AA", "UA"]) & ~both.cancelled & (both.delay > DELAY_THRESHOLD)]
    out["T3"] = {"pct_in_bad_weather": round(100 * sub.bad.mean(), 1),
                 "delayed": int(len(sub)), "delayed_in_bad_weather": int(sub.bad.sum())}

    a = both[both.carrier == "AS"]
    out["T4"] = {"bad_weather_rate": rate(a[a.bad]), "other_rate": rate(a[~a.bad]),
                 "n_bad": int((~a.cancelled & a.bad).sum()), "n_other": int((~a.cancelled & ~a.bad).sum())}

    out["T5"] = {"network_rate": rate(both)}

    counts = both.route.value_counts()
    out["T6"] = {"top5": list(counts.index[:5]), "counts": {r: int(c) for r, c in counts.iloc[:5].items()},
                 "sixth_count": int(counts.iloc[5])}

    b6 = both[(both.carrier == "B6") & (both.date == LOOKUP_DAY)]
    out["T7"] = {"cancelled": int(b6.cancelled.sum())}

    out["T8"] = {"avg_delay_min": round(both[~both.cancelled].delay.mean(), 1)}
    return out


def solve_from_files(data, *, dedup=True, utc_fix=True, wrap=True, strict_weather=True):
    """Reference solver over the written files. Each flag=False reproduces one trap."""
    ap = pd.read_csv(data / "airports.csv").set_index("airport").utc_offset_hours
    wx = pd.read_csv(data / "weather.csv")
    bad = wx.wx_code.isin(BAD_CODES) | ((wx.wx_code == "FG") & (wx.visibility_mi < FOG_VIS_LIMIT))
    if not strict_weather:
        bad = wx.wx_code != "CLR"
    bad_set = {k for k, b in zip(zip(wx.airport, wx.date, wx.hour), bad) if b}

    def mins(s):
        h, m = s.split(":")
        return int(h) * 60 + int(m)

    frames = []
    for cc in CARRIERS:
        f = pd.read_csv(data / "airlines" / f"{cc}.csv", dtype=str, keep_default_na=False)
        if dedup:
            f = f.drop_duplicates("flight_id")
        if "date_utc" in f.columns:
            off = f.origin.map(ap).astype(int) * 60 if utc_fix else 0
            s_utc = pd.to_datetime(f.date_utc + " " + f.sched_dep_utc)
            s_loc = s_utc + pd.to_timedelta(off, unit="min")
            f["date"] = s_loc.dt.strftime("%Y-%m-%d")
            f["sched_dep"] = s_loc.dt.strftime("%H:%M")
            f["actual_dep"] = f.actual_dep_utc  # same shift on both ends -> difference unchanged
            f["sched_ref"] = f.sched_dep_utc
        else:
            f["sched_ref"] = f.sched_dep
        f["carrier"] = cc
        frames.append(f)
    f = pd.concat(frames, ignore_index=True)

    f["cancelled"] = f.cancelled == "true"
    f["sched"] = f.sched_dep.map(mins)
    ref = f.sched_ref.map(mins)
    act = f.actual_dep.map(lambda s: mins(s) if s else None)
    delay = act - ref
    if wrap:
        delay = delay.where(~(delay < -720), delay + 1440)   # departed after midnight
        delay = delay.where(~(delay > 720), delay - 1440)    # early dep. before (UTC) midnight
    f["delay"] = delay
    f["date"] = pd.to_datetime(f.date).dt.date
    f["bad"] = [(o, d.isoformat(), s // 60) in bad_set for o, d, s in zip(f.origin, f.date, f.sched)]
    return answers(f[["carrier", "flight_id", "date", "origin", "dest", "sched", "delay", "cancelled", "bad"]])


# ------------------------------------------------------------------ task text
TASKS = [
    ("T1", "separable",
     "What was the delay rate for each carrier in February 2025? Give a percentage for every carrier.",
     "Six independent per-carrier files: a manager can hand carriers to workers with no cross-talk needed."),
    ("T2", "separable",
     "For each carrier, over January and February 2025: (a) which route had the most scheduled flights "
     "(count cancelled flights too), and (b) which route had the highest delay rate among routes with at "
     "least 20 completed (non-cancelled) flights? Treat a route as an airport pair regardless of direction "
     "(ATL->JFK and JFK->ATL are the same route).",
     "Same per-carrier split as T1 but more work per carrier, so parallel workers save effort."),
    ("T3", "distinct_roles",
     "Consider only Delta (DL), American (AA) and United (UA) flights in January and February 2025. Of their "
     "delayed flights, what percentage were scheduled to depart during bad weather at the origin airport? Use "
     "the weather at the origin during the hour of the scheduled departure.",
     "Needs a weather specialist (codes + visibility rule in weather_codes.md) and a flights specialist; a "
     "generalist juggling all files may use a naive 'anything not clear' rule."),
    ("T4", "distinct_roles",
     "For Alaska (AS) flights in January and February 2025, compare the delay rate of flights scheduled to "
     "depart during bad weather at the origin airport with the delay rate of all other AS flights. Use the "
     "weather at the origin during the hour of the scheduled departure.",
     "Three sources must be combined: AS times are UTC, weather is local, airports.csv holds the offsets. A "
     "timezone/flights role and a weather role each own one boundary."),
    ("T5", "integration",
     "What was the overall delay rate across the whole network (all carriers combined) for January and "
     "February 2025?",
     "If workers return per-carrier percentages, the synthesizer must pool counts, not average the rates."),
    ("T6", "integration",
     "Which 5 routes had the most scheduled flights across all carriers combined in January and February 2025 "
     "(count cancelled flights too)? Treat a route as an airport pair regardless of direction.",
     "Global top-5 is not the union of per-carrier top routes; workers must return full counts for a correct merge."),
    ("T7", "wasteful",
     "How many JetBlue (B6) flights were cancelled on February 14, 2025?",
     "A single lookup in one file: any decomposition is pure overhead."),
    ("T8", "shared_mistake",
     "What was the average departure delay, in minutes, across all completed (non-cancelled) flights by all "
     "carriers in January and February 2025? Early departures count as negative delay.",
     "Every worker runs the same naive time subtraction; post-midnight departures (and B6 duplicates) skew "
     "the mean. Parallel agents can repeat the same mistake; a critic may or may not catch it."),
]

TRAPS = {  # which flags to flip to see what a trap-falling agent would answer
    "no_dedup": dict(dedup=False),
    "no_utc_fix": dict(utc_fix=False),
    "no_midnight_wrap": dict(wrap=False),
    "naive_weather": dict(strict_weather=False),
}


def build(seed, out):
    """Generate one dataset; return (tasks, problems). problems == [] means the dataset is usable."""
    rng = random.Random(seed)
    wx = gen_weather(rng)
    flights = gen_flights(rng, wx)
    write_data(out, flights, wx, rng)

    truth = answers(pd.DataFrame(flights))
    ref = solve_from_files(out)
    assert json.dumps(truth, sort_keys=True) == json.dumps(ref, sort_keys=True), \
        f"reference solver disagrees with ground truth:\n{truth}\n{ref}"

    trap_answers = {name: solve_from_files(out, **flags) for name, flags in TRAPS.items()}
    tasks = []
    for tid, cat, prompt, why in TASKS:
        traps = {name: a[tid] for name, a in trap_answers.items() if a[tid] != truth[tid]}
        tasks.append(dict(id=tid, category=cat, prompt=prompt, why_topology_matters=why,
                          expected=truth[tid], trap_answers=traps))
    # topology-level aggregation traps (what a naive synthesizer would produce)
    df = pd.DataFrame(flights)
    t5 = {cc: rate(df[df.carrier == cc]) for cc in CARRIERS}
    tasks[4]["trap_answers"]["mean_of_carrier_rates"] = {"network_rate": round(sum(t5.values()) / len(t5), 1)}
    df["route"] = [pair(o, d) for o, d in zip(df.origin, df.dest)]
    union = Counter()
    for cc in CARRIERS:
        for r, c in df[df.carrier == cc].route.value_counts().iloc[:3].items():
            union[r] += c
    tasks[5]["trap_answers"]["union_of_carrier_top3"] = {"top5": [r for r, _ in union.most_common(5)]}

    # ---- validity: unambiguous answers, and every trap actually changes its task's answer
    T = {t["id"]: t for t in tasks}
    problems = []
    for cc, v in truth["T2"].items():
        if v["busiest_tie"] or v["worst_tie"]:
            problems.append(f"T2 tie for {cc}")
    if truth["T6"]["counts"][truth["T6"]["top5"][-1]] - truth["T6"]["sixth_count"] < 5:
        problems.append("T6 5th/6th too close")
    if truth["T7"]["cancelled"] < 2:
        problems.append("T7 < 2 cancellations")
    gap = abs(truth["T5"]["network_rate"] - tasks[4]["trap_answers"]["mean_of_carrier_rates"]["network_rate"])
    if gap < 2:
        problems.append(f"T5 mean-of-rates trap too weak ({gap:.1f} pp)")
    if truth["T4"]["n_bad"] < 15:
        problems.append("T4 too few bad-weather AS flights")
    for tid, trap in [("T1", "no_dedup"), ("T1", "no_utc_fix"), ("T3", "naive_weather"), ("T4", "no_utc_fix"),
                      ("T5", "mean_of_carrier_rates"), ("T6", "union_of_carrier_top3"), ("T8", "no_midnight_wrap")]:
        if trap not in T[tid]["trap_answers"]:
            problems.append(f"{tid}: trap {trap} has no effect")
    return flights, tasks, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=853)  # first seed passing all validity checks
    ap.add_argument("--search", action="store_true", help="try seed, seed+1, ... until the dataset is valid")
    ap.add_argument("--out", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--key", default=str(Path(__file__).parent / "tasks.json"))
    args = ap.parse_args()
    out = Path(args.out)

    seed = args.seed
    while True:
        flights, tasks, problems = build(seed, out)
        if not problems or not args.search:
            break
        print(f"seed {seed} rejected: {problems}")
        seed += 1
    if problems:
        print(f"WARNING seed {seed}: {problems}")

    Path(args.key).write_text(json.dumps(dict(seed=seed, delay_threshold=DELAY_THRESHOLD, tasks=tasks),
                                         indent=2, default=str))

    n_mid = sum(1 for f in flights if not f["cancelled"] and f["sched"] + f["delay"] >= 1440)
    print(f"\nseed={seed}  flights={len(flights)} (+{N_DUPS} dup rows in {DUP_CARRIER})  "
          f"cancelled={sum(f['cancelled'] for f in flights)}  past-midnight departures={n_mid}")
    print(f"per carrier: {dict(Counter(f['carrier'] for f in flights))}")
    for t in tasks:
        print(f"\n[{t['id']}] {t['category']}\n  expected: {t['expected']}")
        for name, val in t["trap_answers"].items():
            print(f"  trap {name:24s}-> {val}")


if __name__ == "__main__":
    main()
