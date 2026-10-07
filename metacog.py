"""
metacog.py (v2): Metacognitive monitor for a persistent agent (muse_metacog; agent-loop blueprint steps 6-7).

Pieces:
  1. Scoring:      brier (+ Murphy decomposition), ece, auroc2, dprime, fit_meta_d (MLE, Maniscalco & Lau 2012)
  2. Calibration:  IsotonicCalibrator (PAV) mapping a fused raw score -> calibrated p
  1b. murphy_summary(): Brier skill + reliability/resolution/uncertainty diagnosis for the grader
  3. Fusion:       Fusion: prior weights, fit() by L2-to-prior logistic regression on logged outcomes
  3b. MonitorPolicy: which signals to compute per claim (none/cheap/mid/full) and what they cost
  4. Control:      EscalationGate: cost-based System 1 -> System 2 handoff
  5. Logging:      OutcomeLog: append-only JSONL of (raw signals, p, outcome) for refitting + reports

Requires: numpy, scipy
"""
from __future__ import annotations
import json, math, time
from dataclasses import dataclass, field, asdict
from typing import Sequence
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# ---------------------------------------------------------------- 1. scoring
def brier(p: Sequence[float], y: Sequence[int], n_bins: int = 10) -> dict:
    """Brier score with Murphy decomposition: BS ~= reliability - resolution + uncertainty."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    bs = float(np.mean((p - y) ** 2))
    base = y.mean()
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    rel = res = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            w = m.mean()
            rel += w * (p[m].mean() - y[m].mean()) ** 2
            res += w * (y[m].mean() - base) ** 2
    return {"brier": bs, "reliability": rel, "resolution": res, "uncertainty": base * (1 - base)}

def murphy_summary(p, y, n_bins: int = 10) -> dict:
    """
    Grader-ready diagnosis. Brier skill score vs climatology (always predicting the base rate),
    plus WHY the Brier is what it is:
      reliability high  -> miscalibration (fixable by recalibration)
      resolution low    -> no discrimination (signals carry little information)
      uncertainty high  -> hard base rate (near 50/50), not the agent's fault
    """
    d = brier(p, y, n_bins)
    unc = d["uncertainty"]
    bss = 1 - d["brier"] / unc if unc > 0 else float("nan")
    rel_share = d["reliability"] / d["brier"] if d["brier"] > 0 else 0.0
    res_ratio = d["resolution"] / unc if unc > 0 else 0.0
    if rel_share > 0.15:
        dx = "miscalibration: recalibrate (isotonic) before touching signals"
    elif res_ratio < 0.1:
        dx = "low discrimination: confidence barely separates right from wrong; improve signals/fusion"
    elif unc > 0.2 and bss < 0.1:
        dx = "hard base rate: near-coin-flip domain, low skill expected"
    else:
        dx = "healthy"
    return {**{k: round(float(v), 4) for k, v in d.items()},
            "brier_skill": round(float(bss), 4), "reliability_share": round(float(rel_share), 3),
            "resolution_ratio": round(float(res_ratio), 3), "diagnosis": dx, "n": len(p)}

def ece(p: Sequence[float], y: Sequence[int], n_bins: int = 15) -> float:
    """Expected Calibration Error, equal-width bins. Not a proper scoring rule; report beside Brier."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    return float(sum((bins == b).mean() * abs(y[bins == b].mean() - p[bins == b].mean())
                     for b in range(n_bins) if (bins == b).any()))

def auroc2(conf: Sequence[float], correct: Sequence[int]) -> float:
    """Type-2 AUROC = P(conf|correct > conf|incorrect), ties count 0.5 (Mann-Whitney)."""
    c, k = np.asarray(conf, float), np.asarray(correct, bool)
    pos, neg = c[k], c[~k]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))

def _rates(stim, resp):
    stim, resp = np.asarray(stim), np.asarray(resp)
    # log-linear correction (Hautus 1995) avoids infinite z
    H = ((resp[stim == 1] == 1).sum() + 0.5) / ((stim == 1).sum() + 1)
    FA = ((resp[stim == 0] == 1).sum() + 0.5) / ((stim == 0).sum() + 1)
    return H, FA

def dprime(stim, resp) -> tuple[float, float]:
    """Type-1 d' and criterion c for a binary task (stim, resp in {0,1})."""
    H, FA = _rates(stim, resp)
    return float(norm.ppf(H) - norm.ppf(FA)), float(-0.5 * (norm.ppf(H) + norm.ppf(FA)))

def fit_meta_d(stim, resp, rating, n_ratings: int) -> dict:
    """
    MLE meta-d' (Maniscalco & Lau 2012), equal-variance SDT.
    stim, resp in {0,1}; rating in {1..n_ratings} (1 = lowest confidence).
    Only valid for a genuine binary type-1 decision (e.g. 'is this claim true?', 'will this tool call succeed?').
    Returns d', meta-d', M-ratio, M-diff.
    """
    stim, resp, rating = map(np.asarray, (stim, resp, rating))
    d, c = dprime(stim, resp)
    K = n_ratings
    # counts[s, r, k]
    counts = np.zeros((2, 2, K))
    for s, r, k in zip(stim, resp, rating):
        counts[s, r, k - 1] += 1
    counts += 1.0 / (2 * K)  # small padding for stability
    c_prime = c / d if abs(d) > 1e-6 else 0.0  # criterion normalised by d'

    def unpack(theta):
        md = theta[0]
        mc = c_prime * md
        inc = np.exp(theta[1:])
        up = mc + np.cumsum(inc[: K - 1])           # criteria above mc  (resp=1 side)
        dn = mc - np.cumsum(inc[K - 1:])            # criteria below mc  (resp=0 side)
        return md, mc, up, dn

    def nll(theta):
        md, mc, up, dn = unpack(theta)
        ll = 0.0
        for s in (0, 1):
            mu = md / 2 if s == 1 else -md / 2
            F = lambda x: norm.cdf(x - mu)
            # resp = 1: bins [mc, up1), [up1, up2) ... [up_{K-1}, inf)
            edges1 = np.concatenate(([mc], up, [np.inf]))
            p1 = np.diff(F(edges1)); p1 /= max(1 - F(mc), 1e-12)
            # resp = 0: bins (dn1, mc], (dn2, dn1] ... (-inf, dn_{K-1}] ; rating 1 nearest mc
            edges0 = np.concatenate(([mc], dn, [-np.inf]))
            p0 = -np.diff(F(edges0)); p0 /= max(F(mc), 1e-12)
            ll += (counts[s, 1] * np.log(np.clip(p1, 1e-12, None))).sum()
            ll += (counts[s, 0] * np.log(np.clip(p0, 1e-12, None))).sum()
        return -ll

    x0 = np.concatenate(([d], np.log(np.full(2 * (K - 1), 0.5))))
    fit = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-8})
    md = float(fit.x[0])
    return {"d_prime": d, "meta_d": md, "m_ratio": md / d if d else float("nan"), "m_diff": md - d,
            "converged": bool(fit.success)}

# ---------------------------------------------------------------- 2. calibration
class IsotonicCalibrator:
    """Pool-adjacent-violators isotonic regression: raw score -> calibrated probability."""
    def __init__(self):
        self.x = np.array([0.0, 1.0]); self.y = np.array([0.0, 1.0])

    def fit(self, raw, y):
        o = np.argsort(raw); x = np.asarray(raw, float)[o]; v = np.asarray(y, float)[o]
        blocks = [[v[i], 1.0, x[i], x[i]] for i in range(len(v))]  # mean, weight, xmin, xmax
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][0] > blocks[i + 1][0]:
                a, b = blocks[i], blocks.pop(i + 1)
                w = a[1] + b[1]
                blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / w, w, a[2], b[3]]
                i = max(i - 1, 0)
            else:
                i += 1
        self.x = np.array([(b[2] + b[3]) / 2 for b in blocks])
        self.y = np.clip(np.array([b[0] for b in blocks]), 0.01, 0.99)
        return self

    def __call__(self, raw):
        return np.interp(raw, self.x, self.y)

# ---------------------------------------------------------------- 3. signal fusion
@dataclass
class Signals:
    """Any signal may be None when the monitor tier skipped it (see MonitorPolicy)."""
    verbal_conf: float                       # stated confidence 0..1 (Tian 2023 / Xiong 2024): ~free
    retrieval_support: float | None = None   # 0..1 share of retrieved evidence entailing the claim
    p_true: float | None = None              # P(True) self-evaluation pass (Kadavath 2022)
    semantic_entropy: float | None = None    # nats over meaning clusters (Farquhar 2024); 0 = all agree
    n_samples: int = 0                       # N used for semantic entropy

SIGNAL_NAMES = ("agree", "p_true", "verbal_conf", "retrieval_support")
PRIOR_W = np.array([0.4, 0.2, 0.1, 0.3])   # asserted prior (Farquhar: SE > self-report); NOT fitted

def _features(s: Signals):
    """Feature vector + mask of which signals are present."""
    agree = None
    if s.semantic_entropy is not None and s.n_samples >= 2:
        agree = 1 - min(s.semantic_entropy / math.log(s.n_samples), 1.0)
    v = [agree, s.p_true, s.verbal_conf, s.retrieval_support]
    mask = np.array([x is not None for x in v])
    return np.array([x if x is not None else 0.0 for x in v], float), mask

@dataclass
class Fusion:
    """
    Linear fusion with missing-signal renormalisation. Starts at the asserted prior; call fit()
    on logged outcomes once there is volume. Fitting is logistic regression with an L2 penalty
    pulling weights toward the prior (strength `l2`), so small n stays near the prior and large n
    lets the data speak. Isotonic calibration still runs downstream: it fixes monotone
    miscalibration but cannot recover information a bad weighting destroys, hence this fit.
    """
    w: np.ndarray = field(default_factory=lambda: PRIOR_W.copy())
    b: float = 0.0
    fitted: bool = False
    n_fit: int = 0

    def raw(self, s: Signals) -> float:
        x, m = _features(s)
        if not self.fitted:   # prior mode: weighted mean of present signals, stays in [0,1]
            w = self.w * m
            return float(np.dot(w, x) / w.sum()) if w.sum() > 0 else 0.5
        z = self.b + np.dot(self.w * m, x) * (self.w.sum() / max((self.w * m).sum(), 1e-9))
        return float(1 / (1 + math.exp(-z)))

    def fit(self, records: list[dict], l2: float = 5.0, min_n: int = 100) -> "Fusion":
        rows = [r for r in records if r.get("correct") is not None]
        if len(rows) < min_n:
            return self                       # not enough volume: keep the prior
        X, M, y = [], [], []
        for r in rows:
            x, m = _features(Signals(**{k: r.get(k) for k in Signals.__dataclass_fields__ if k in r}))
            X.append(x); M.append(m); y.append(r["correct"])
        X, M, y = np.array(X), np.array(M), np.array(y, float)
        # prior expressed on the logit scale: scale so prior mean output spans roughly +-3 logits
        w0 = PRIOR_W * 6.0; b0 = -3.0
        def loss(theta):
            b, w = theta[0], theta[1:]
            z = b + ((X * M) @ w) * (w.sum() / np.clip((M * w).sum(1), 1e-9, None))
            ll = np.sum(y * z - np.logaddexp(0, z))
            return -ll + l2 * (np.sum((w - w0) ** 2) + (b - b0) ** 2)
        res = minimize(loss, np.concatenate(([b0], w0)), method="L-BFGS-B",
                       bounds=[(None, None)] + [(0, None)] * 4)   # signals assumed non-negative evidence
        self.b, self.w, self.fitted, self.n_fit = float(res.x[0]), res.x[1:], True, len(rows)
        return self

    def describe(self) -> dict:
        tot = self.w.sum() or 1
        return {"fitted": self.fitted, "n_fit": self.n_fit,
                "relative_weights": dict(zip(SIGNAL_NAMES, np.round(self.w / tot, 3).tolist()))}

def fuse_signals(s: Signals, fusion: Fusion | None = None) -> float:
    """Back-compat wrapper. Raw (uncalibrated) score; feed through IsotonicCalibrator."""
    return (fusion or Fusion()).raw(s)

# ---------------------------------------------------------------- 3b. monitor cost policy
TIER_COST = {   # inference units per claim; tune to your stack
    "verbal_conf": 0.0,          # comes with the answer
    "retrieval_support": 1.5,    # retrieval + entailment check
    "p_true": 1.0,               # one self-evaluation pass
    "semantic_entropy": 10.0,    # N=10 samples + clustering
}

@dataclass
class MonitorPolicy:
    """
    Decides which signals to compute BEFORE computing them, because the monitor has its own cost.
      full  : all signals             (internal bets, high-stakes or irreversible actions)
      mid   : verbal + retrieval + P(True)
      cheap : verbal + retrieval      (default per-turn monitoring)
      none  : verbal only             (chit-chat, trivially reversible)
    Escalate the tier itself if the cheap signals disagree (|verbal - retrieval| > disagree).
    """
    full_cost_error: float = 20.0
    mid_cost_error: float = 5.0
    disagree: float = 0.35
    budget_per_turn: float = 3.0   # inference units allowed outside 'full'

    TIERS = {"none": ("verbal_conf",),
             "cheap": ("verbal_conf", "retrieval_support"),
             "mid": ("verbal_conf", "retrieval_support", "p_true"),
             "full": ("verbal_conf", "retrieval_support", "p_true", "semantic_entropy")}

    def choose(self, cost_error: float, internal_bet: bool = False, needs_facts: bool = True) -> str:
        if internal_bet or cost_error >= self.full_cost_error:
            return "full"
        if cost_error >= self.mid_cost_error:
            return "mid"
        return "cheap" if needs_facts else "none"

    def maybe_upgrade(self, tier: str, partial: Signals) -> str:
        """After cheap signals arrive: upgrade one step if they disagree and budget allows."""
        if tier in ("cheap", "none") and partial.retrieval_support is not None \
                and abs(partial.verbal_conf - partial.retrieval_support) > self.disagree:
            return "mid" if self.cost("mid") <= self.budget_per_turn else tier
        return tier

    def cost(self, tier: str) -> float:
        return sum(TIER_COST[s] for s in self.TIERS[tier])

# ---------------------------------------------------------------- 4. escalation gate
@dataclass
class EscalationGate:
    """
    Escalate to System 2 when expected cost of acting now exceeds the cost of verifying:
        (1 - p) * cost_error  >  cost_verify + (1 - verify_recall) * (1 - p) * cost_error
    =>  escalate if p < 1 - cost_verify / (verify_recall * cost_error)
    System 2 MUST use external grounding (tools, tests, retrieval, user): intrinsic
    self-critique does not reliably fix errors (Huang et al., ICLR 2024).
    """
    cost_verify: float = 1.0
    verify_recall: float = 0.8   # share of errors System 2 actually catches (measure it!)
    surprise_override: float = 2.0  # escalate regardless if prediction-error z-score exceeds this

    def threshold(self, cost_error: float) -> float:
        return max(0.0, 1 - self.cost_verify / (self.verify_recall * cost_error))

    def decide(self, p: float, cost_error: float, surprise_z: float = 0.0) -> dict:
        tau = self.threshold(cost_error)
        esc = p < tau or surprise_z > self.surprise_override
        return {"escalate": esc, "p": p, "tau": tau,
                "reason": "surprise" if surprise_z > self.surprise_override else ("low_p" if p < tau else "ok")}

# ---------------------------------------------------------------- 5. outcome log + report
@dataclass
class OutcomeLog:
    path: str
    def write(self, domain: str, signals: Signals, raw: float, p: float, escalated: bool,
              correct: int | None = None, binary: dict | None = None,
              tier: str | None = None, monitor_cost: float | None = None):
        """`correct` must come from external ground truth (grader, tool record, user), never self-report."""
        rec = {"t": time.time(), "domain": domain, **asdict(signals), "raw": raw, "p": p,
               "escalated": escalated, "correct": correct, "binary": binary,
               "tier": tier, "monitor_cost": monitor_cost}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def load(self):
        with open(self.path) as f:
            return [json.loads(l) for l in f if l.strip()]

def report(records: list[dict]) -> dict:
    """Per-domain calibration + sensitivity + gate control metrics (the self-model's capability map)."""
    out = {}
    for dom in sorted({r["domain"] for r in records}):
        rs = [r for r in records if r["domain"] == dom and r["correct"] is not None]
        if len(rs) < 10:
            continue
        p = [r["p"] for r in rs]; y = [r["correct"] for r in rs]
        errs = [r for r in rs if not r["correct"]]
        out[dom] = {"accuracy": float(np.mean(y)), **murphy_summary(p, y), "ece": ece(p, y),
                    "auroc2": auroc2(p, y),
                    "gate_error_recall": float(np.mean([r["escalated"] for r in errs])) if errs else None,
                    "escalation_rate": float(np.mean([r["escalated"] for r in rs]))}
        bins = [r["binary"] for r in rs if r.get("binary")]
        if len(bins) >= 50:
            out[dom]["meta_d"] = fit_meta_d([b["stim"] for b in bins], [b["resp"] for b in bins],
                                            [b["rating"] for b in bins], n_ratings=4)
    return out

# ---------------------------------------------------------------- demo / self-test
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # meta-d' sanity check: ideal observer (M ~ 1) vs observer with noisy confidence (M < 1)
    n, d_true, K = 6000, 1.5, 4
    stim = rng.integers(0, 2, n)
    x1 = rng.normal(np.where(stim == 1, d_true / 2, -d_true / 2), 1)
    resp = (x1 > 0).astype(int)
    for label, sd in (("ideal", 0.0), ("noisy-confidence", 1.0)):
        x2 = x1 + rng.normal(0, 1, n) * sd
        strength = np.where(resp == 1, x2, -x2)          # evidence toward the chosen response
        rating = np.digitize(strength, [0.4, 0.9, 1.5]) + 1
        fit = fit_meta_d(stim, resp, rating, K)
        print(f"{label:>17}: d'={fit['d_prime']:.2f} meta-d'={fit['meta_d']:.2f} M-ratio={fit['m_ratio']:.2f}")

    # calibration + gate demo: overconfident raw scores
    raw = rng.uniform(0.3, 1.0, 2000)
    y = (rng.uniform(size=2000) < raw ** 2).astype(int)   # true reliability lower than raw
    print("raw  :", {k: round(float(v), 4) for k, v in brier(raw, y).items()}, "ECE", round(ece(raw, y), 4))
    cal = IsotonicCalibrator().fit(raw[:1000], y[:1000])
    p = cal(raw[1000:])
    print("calib:", {k: round(float(v), 4) for k, v in brier(p, y[1000:]).items()}, "ECE", round(ece(p, y[1000:]), 4),
          "AUROC2", round(auroc2(p, y[1000:]), 4))

    print("murphy:", murphy_summary(raw[1000:], y[1000:])["diagnosis"], "->",
          murphy_summary(p, y[1000:])["diagnosis"])

    # fusion fit on synthetic logs: true signal quality differs from the prior's assumption
    recs = []
    for _ in range(1500):
        truth = rng.uniform() < 0.6
        se = max(0.0, rng.normal(0.8 if truth else 1.0, 0.5))       # weak, despite the prior's 0.4
        recs.append({"verbal_conf": float(np.clip(rng.normal(0.85, 0.1), 0, 1)),   # uninformative
                     "retrieval_support": float(np.clip(rng.normal(0.7 if truth else 0.4, 0.2), 0, 1)),  # strong
                     "p_true": float(np.clip(rng.normal(0.7 if truth else 0.6, 0.15), 0, 1)),
                     "semantic_entropy": se, "n_samples": 10, "correct": int(truth)})
    fz = Fusion().fit(recs[:1000])
    print("fusion:", fz.describe())
    for name, f in (("prior", Fusion()), ("fitted", fz)):
        r = [f.raw(Signals(**{k: x[k] for k in Signals.__dataclass_fields__})) for x in recs[1000:]]
        print(f"  {name:>6} AUROC2={auroc2(r, [x['correct'] for x in recs[1000:]]):.3f}")

    # monitor policy + gate
    pol, gate = MonitorPolicy(), EscalationGate()
    for label, ce, bet in (("chat reply", 1, False), ("calendar edit", 6, False), ("internal bet", 2, True)):
        t = pol.choose(ce, internal_bet=bet)
        print(f"{label:>13}: tier={t:<5} monitor_cost={pol.cost(t):>4.1f}")
    partial = Signals(verbal_conf=0.95, retrieval_support=0.3)
    print("cheap signals disagree -> upgrade to", pol.maybe_upgrade("cheap", partial))
    cal2 = IsotonicCalibrator().fit([fz.raw(Signals(**{k: x[k] for k in Signals.__dataclass_fields__}))
                                     for x in recs[:1000]], [x["correct"] for x in recs[:1000]])
    pc = float(cal2(fz.raw(partial)))
    print("gate:", gate.decide(pc, cost_error=6))
