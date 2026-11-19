"""GTA VI trophy rarity forecaster (make.fun: gta6-trophy-rarity-forecast-01).

Reads every case file (makefun/gta6-trophies-case/v1) from the input
directory and writes five quantiles of each trophy's earn rate at the
case target to the output directory, one <id>.out per <id>.in.

Model, built from the competition's public rules and the case data:
  * Counts first, not rates.  A read's rate inverts exactly to an
    integer earner count (count = round(rate * players / 100)); the
    counts sum back to the read's trophies_earned.  Earners and players
    are modelled as two separate series and the rate is their ratio.
  * A read shows the page as of Exophase's last refresh.  Identical
    consecutive reads are one stale snapshot; a change is a refresh.
    Between refreshes the true counts keep moving while the page sits
    still, so a stale page is not zero growth.
  * At a refresh the denominator jumps (a new cohort of tracked
    players) while most earners barely move, so rates fall.  Trophies
    the new cohort earns immediately stay saturated (the early fishing
    trophy is at 100% in every read and stays there); the rest grow
    slowly, faster for early trophies, slower for late/grind ones.
  * Growth rates are fitted from the snapshots in the case history
    (recent snapshots weigh more) and damped over long horizons.
    With a flat history the launch-night priors apply: the page sat
    unchanged for hours on launch night, then refreshed with players
    up roughly six-fold while non-early earners grew about a third.
  * Quantiles come from the projected median plus a spread that
    combines parameter, horizon, count and refresh-timing uncertainty.

Standard library only.  Deterministic: no randomness, no clock reads,
no network.  Every case always produces a valid forecast (a
persistence fallback covers degenerate inputs).
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from statistics import NormalDist

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
_NORM = NormalDist()
ZTAU = [_NORM.inv_cdf(t) for t in LEVELS]

# Launch-night priors (from the public competition rules, which describe
# the first refresh of launch night: the page unchanged for hours, then
# players tracked up about six-fold while the early fishing trophy stayed
# at 100% and other earners grew roughly a third).
PRIOR_REFRESH_RATIO = 6.0
PRIOR_REFRESH_INTERVAL_H = 4.5
PRIOR_EARNER_GROWTH = 1.31
PRIOR_DECAY_TAU_H = 72.0

SPREAD_BASE = 0.10
SPREAD_HORIZON_K = 0.05
TIMING_UNCERTAINTY_H = 0.25


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _clamp_rate(x):
    return min(99.99, max(0.01, x))


def _logit(pct):
    p = _clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def _inv_logit(logit_value):
    return 100.0 / (1.0 + math.exp(-logit_value))


def _median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def _weighted_median(pairs):
    """pairs: list of (value, weight). Returns the weighted median."""
    items = sorted(pairs)
    total = sum(w for _, w in items)
    if total <= 0:
        return _median([v for v, _ in items])
    acc = 0.0
    for value, weight in items:
        acc += weight
        if acc >= 0.5 * total:
            return value
    return items[-1][0]


def _early_score(trophy):
    """How early/easy a trophy is, from its metadata (0 = late grind,
    1 = earned immediately). Drives growth weighting and shrinkage."""
    desc = (trophy.get("description") or "").lower()
    tier = (trophy.get("tier") or "").lower()
    if "gone fishing" in desc:
        return 1.0
    if "chapter 1" in desc:
        return 0.9
    if "chapter 2" in desc:
        return 0.7
    if "chapter 3" in desc:
        return 0.5
    if "chapter 4" in desc:
        return 0.4
    if "final mission" in desc:
        return 0.25
    if "100% completion" in desc:
        return 0.05
    if tier == "platinum":
        return 0.05
    if tier == "gold":
        return 0.15
    numbers = [int(n) for n in re.findall(r"\d+", desc)]
    big = [n for n in numbers if n >= 50]
    if big:
        return 0.15
    mid = [n for n in numbers if 20 <= n < 50]
    if mid:
        return 0.25
    small = [n for n in numbers if 8 <= n < 20]
    if small:
        return 0.35
    if any(w in desc for w in ("arcade", "meditate", "cocktail",
                               "bowling", "fishing", "activity")):
        return 0.7
    return 0.45


def _damped_time(hours, tau=PRIOR_DECAY_TAU_H):
    """Effective growth time with exponential decay of the growth rate:
    integral of exp(-t/tau) from 0 to hours."""
    if hours <= 0:
        return 0.0
    return tau * (1.0 - math.exp(-hours / tau))


def _reconstruct_counts(case):
    """Turn each history read into (time, players, [earners per trophy]).
    Returns (reads, trophies) with reads sorted oldest first."""
    trophies = case.get("trophies", [])
    reads = []
    for entry in case.get("history", []):
        players = entry.get("players_tracked")
        rates = entry.get("rates") or []
        if players is None or not rates:
            continue
        earners = []
        for rate in rates:
            if rate is None:
                earners.append(None)
            else:
                earners.append(int(round(rate * players / 100.0)))
        try:
            when = _parse_time(entry.get("at") or entry.get("slot"))
        except (ValueError, TypeError):
            continue
        reads.append({"at": when, "players": players, "earners": earners})
    reads.sort(key=lambda r: r["at"])
    return reads, trophies


def _segment_snapshots(reads, n_trophies):
    """Group consecutive identical reads into snapshots. A snapshot is
    the page as shown between two refreshes."""
    snapshots = []
    for read in reads:
        key = (read["players"], tuple(read["earners"]))
        if snapshots and snapshots[-1]["key"] == key:
            snapshots[-1]["end"] = read["at"]
            snapshots[-1]["reads"] += 1
        else:
            snapshots.append({
                "key": key,
                "start": read["at"],
                "end": read["at"],
                "players": read["players"],
                "earners": list(read["earners"]),
                "reads": 1,
            })
    return snapshots


def _fit_growth(snapshots, trophies, early):
    """Fit per-hour log growth rates from snapshot-to-snapshot changes.
    Returns (lambda_players, [lambda per trophy], interval_h,
    residual_std). Falls back to the launch-night priors when the
    history is flat."""
    n = len(trophies)
    interval = PRIOR_REFRESH_INTERVAL_H
    lambda_p = math.log(PRIOR_REFRESH_RATIO) / PRIOR_REFRESH_INTERVAL_H
    lambda_e = [math.log(PRIOR_EARNER_GROWTH) / PRIOR_REFRESH_INTERVAL_H] * n
    residual_std = 0.0
    if len(snapshots) < 2:
        return lambda_p, lambda_e, interval, residual_std

    pairs = []
    for a, b in zip(snapshots, snapshots[1:]):
        dt = (b["start"] - a["start"]).total_seconds() / 3600.0
        if dt <= 0:
            continue
        pairs.append((a, b, dt))
    if not pairs:
        return lambda_p, lambda_e, interval, residual_std

    last_start = snapshots[-1]["start"]
    interval = _median([(b["start"] - a["start"]).total_seconds() / 3600.0
                        for a, b, _ in pairs])
    if interval <= 0:
        interval = PRIOR_REFRESH_INTERVAL_H

    def recency_weight(end_time):
        age_h = (last_start - end_time).total_seconds() / 3600.0
        return 0.5 ** max(age_h, 0.0) / 24.0

    lp_items = []
    for a, b, dt in pairs:
        if a["players"] > 0 and b["players"] > 0:
            lp_items.append((math.log(b["players"] / a["players"]) / dt,
                             recency_weight(b["start"])))
    if lp_items:
        lambda_p = _weighted_median(lp_items)
    lambda_p = min(1.0, max(-0.01, lambda_p))

    raw_lambda = []
    for i in range(n):
        items = []
        for a, b, dt in pairs:
            ea, eb = a["earners"][i], b["earners"][i]
            if ea is None or eb is None or ea <= 0 or eb <= 0:
                continue
            items.append((min(1.5, max(-0.02, math.log(eb / ea) / dt)),
                          recency_weight(b["start"])))
        raw_lambda.append(items)
    all_raw = [v for items in raw_lambda for v, _ in items]
    global_lambda = _median(all_raw) if all_raw else lambda_e[0]
    global_lambda = min(1.0, max(0.0, global_lambda))
    prior_lambda = [math.log(PRIOR_EARNER_GROWTH) / PRIOR_REFRESH_INTERVAL_H
                    * (0.4 + 1.2 * e) for e in early]
    fitted = []
    for i in range(n):
        items = raw_lambda[i]
        if items:
            raw = _weighted_median(items)
            count = len(items)
            fitted.append((count * raw + 2.0 * global_lambda
                           + 1.0 * prior_lambda[i]) / (count + 3.0))
        else:
            fitted.append(0.5 * global_lambda + 0.5 * prior_lambda[i])
    lambda_e = [min(1.5, max(0.0, v)) for v in fitted]

    residuals = []
    for a, b, dt in pairs:
        for i in range(n):
            ea, eb = a["earners"][i], b["earners"][i]
            if ea is None or eb is None or ea <= 0 or eb <= 0:
                continue
            observed = math.log(eb / ea)
            predicted = (lambda_e[i] - lambda_p) * dt
            residuals.append(observed - predicted)
    if len(residuals) >= 4:
        mean_res = sum(residuals) / len(residuals)
        var = sum((r - mean_res) ** 2 for r in residuals) / len(residuals)
        residual_std = math.sqrt(var)
    return lambda_p, lambda_e, interval, residual_std


def _attainment_prior(early):
    out = []
    for e in early:
        if e >= 0.95:
            out.append(1.0)
        elif e >= 0.7:
            out.append(0.01)
        elif e >= 0.4:
            out.append(0.003)
        else:
            out.append(0.001)
    return out


def forecast_case(case):
    """Return {trophy id: [q05, q25, q50, q75, q95]} for one case."""
    trophies = case.get("trophies", [])
    levels = case.get("levels") or LEVELS
    zvals = [_NORM.inv_cdf(t) for t in levels]
    if not trophies:
        return {}
    early = [_early_score(t) for t in trophies]
    try:
        target = _parse_time(case["target"])
        checkpoint = _parse_time(case["at"])
    except (KeyError, ValueError, TypeError):
        target = None
        checkpoint = None
    reads, _ = _reconstruct_counts(case)
    if not reads or target is None:
        return _fallback_forecast(trophies, levels, zvals)

    snapshots = _segment_snapshots(reads, len(trophies))
    last = snapshots[-1]
    ref_players = last["players"]
    ref_earners = [0 if e is None else e for e in last["earners"]]
    t_ref = last["start"]

    lambda_p, lambda_e, interval, resid_std = _fit_growth(
        snapshots, trophies, early)
    attainment = _attainment_prior(early)
    flat = len(snapshots) < 2

    # Predict the refresh times after the last snapshot; the answer read
    # shows the last refresh at or before the target.
    t_last = t_ref
    nxt = t_ref
    limit = target.timestamp()
    while True:
        cand = nxt.timestamp() + interval * 3600.0
        if cand > limit:
            break
        nxt = datetime.fromtimestamp(cand, tz=timezone.utc)
        t_last = nxt
    delta_h = max((t_last - t_ref).total_seconds() / 3600.0, 0.0)
    g_time = _damped_time(delta_h)
    if g_time <= 0:
        g_time = max(delta_h, 1.0)

    if flat:
        # No refresh observed: apply the launch-night priors through the
        # mechanism (players up ~6x at the refresh, earners up ~31%
        # weighted by earliness, the saturated trophy stays saturated).
        # Rates are defined so the projection reproduces those factors
        # exactly over this case's window.
        weights = [0.9 + 0.2 * e for e in early]
        unsat = [i for i in range(len(trophies))
                 if not (ref_players > 0
                         and ref_earners[i] >= ref_players)]
        wmean = 1.0
        if unsat:
            total = sum(ref_earners[i] for i in unsat)
            if total > 0:
                wmean = sum(ref_earners[i] * weights[i]
                            for i in unsat) / total
        lambda_p = math.log(PRIOR_REFRESH_RATIO) / g_time
        lambda_e = []
        for i, trophy in enumerate(trophies):
            if ref_players > 0 and ref_earners[i] >= ref_players:
                lambda_e.append(lambda_p)
            else:
                growth = 1.0 + (PRIOR_EARNER_GROWTH - 1.0) * weights[i] / wmean
                lambda_e.append(math.log(max(growth, 1.0)) / g_time)

    players_ans = ref_players * math.exp(lambda_p * g_time)
    players_ans = min(players_ans, ref_players * 50.0 + 1.0)
    players_ans = max(players_ans, 1.0)

    horizon_days = max(delta_h, 0.0) / 24.0
    result = {}
    for i, trophy in enumerate(trophies):
        tid = trophy["id"]
        earners_ans = ref_earners[i] * math.exp(lambda_e[i] * g_time)
        if ref_earners[i] == 0:
            # Nobody has it yet: seed from the incoming cohort only.
            earners_ans += attainment[i] * max(players_ans - ref_players, 0.0)
        earners_ans = min(earners_ans, players_ans)
        earners_ans = max(earners_ans, 0.0)
        median = 100.0 * earners_ans / players_ans
        median = _clamp_rate(median)
        # Safety rail: over one projection a rate cannot credibly rise
        # more than ~4.5x or fall more than ~20x.  (Exponential
        # extrapolation of a fast launch-day rate over days would
        # otherwise saturate early trophies at 100%.)
        if ref_players > 0 and ref_earners[i] > 0:
            last_rate = 100.0 * ref_earners[i] / ref_players
            median = min(median, _clamp_rate(last_rate * math.exp(1.5)))
            median = max(median, _clamp_rate(last_rate * math.exp(-3.0)))
        # Saturated trophies (everyone tracked has it) stay saturated,
        # but the lower quantiles hedge the chance that a few of the
        # newly tracked players lack it (the public rules say the rate
        # stayed at 100% at the first refresh; the hedge costs little
        # if so and a lot if not).
        saturated = ref_players > 0 and ref_earners[i] >= ref_players
        if saturated:
            median = 99.99
        spread = math.sqrt(
            SPREAD_BASE ** 2
            + (SPREAD_HORIZON_K * math.sqrt(horizon_days)) ** 2
            + min(0.08, 0.25 / math.sqrt(earners_ans + 1.0)) ** 2
            + (abs(lambda_e[i] - lambda_p) * TIMING_UNCERTAINTY_H) ** 2
            + (resid_std * math.sqrt(max(g_time, 1.0) / 24.0)
               if resid_std > 0 else 0.0) ** 2
        )
        spread = min(1.0, max(0.08, spread))
        base = _logit(median)
        quantiles = []
        for z in zvals:
            q = _inv_logit(base + z * spread)
            quantiles.append(round(_clamp_rate(q), 4))
        if saturated:
            quantiles[0] = 99.9
            if len(quantiles) > 1:
                quantiles[1] = 99.9
            for k in range(2, len(quantiles)):
                quantiles[k] = 99.99
        for k in range(1, len(quantiles)):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        result[tid] = quantiles
    return result


def _fallback_forecast(trophies, levels, zvals):
    """Persistence with a random-walk spread; only used when the case
    has no usable history at all. Keeps the entry valid."""
    out = {}
    prior = {"platinum": 0.1, "gold": 0.2, "silver": 0.5, "bronze": 1.0}
    for trophy in trophies:
        rate = prior.get((trophy.get("tier") or "").lower(), 1.0)
        base = _logit(rate)
        quantiles = []
        for z in zvals:
            quantiles.append(round(_clamp_rate(_inv_logit(base + z * 1.0)), 4))
        for k in range(1, len(quantiles)):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        out[trophy["id"]] = quantiles
    return out


def _persistence_forecast(case, levels, zvals):
    """Last-rate median with a random-walk spread (house-baseline shape),
    used only if the main model fails on a case."""
    reads, _ = _reconstruct_counts(case)
    trophies = case.get("trophies", [])
    out = {}
    if not reads:
        return _fallback_forecast(trophies, levels, zvals)
    last = reads[-1]
    horizon_days = max(case.get("horizon_minutes", 90), 0) / 1440.0
    half = 0.5 * math.sqrt(horizon_days)
    for i, trophy in enumerate(trophies):
        rate = last["earners"][i]
        if rate is None:
            rate = 1.0
        else:
            rate = rate / last["players"] * 100.0 if last["players"] else 1.0
        base = _logit(rate)
        quantiles = []
        for z in zvals:
            quantiles.append(round(_clamp_rate(_inv_logit(base + z * half)), 4))
        for k in range(1, len(quantiles)):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        out[trophy["id"]] = quantiles
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        path = os.path.join(in_dir, name)
        with open(path, "r", encoding="utf-8") as handle:
            case = json.load(handle)
        levels = case.get("levels") or LEVELS
        zvals = [_NORM.inv_cdf(t) for t in levels]
        try:
            forecast = forecast_case(case)
            if not forecast:
                raise ValueError("empty forecast")
        except Exception:
            forecast = _persistence_forecast(case, levels, zvals)
        rows = []
        for trophy in case.get("trophies", []):
            tid = trophy["id"]
            rows.append({"id": tid, "q": forecast[tid]})
        payload = {"trophies": rows}
        out_path = os.path.join(out_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
