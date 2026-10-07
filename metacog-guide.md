# Wiring `metacog.py` into an agent loop

`metacog.py` covers steps 6, 7 and 10 of the [cognitive cycle](architecture.md): monitor a draft, decide whether to escalate, log the outcome, and refit offline.

## Per-claim loop

```python
from metacog import (Signals, Fusion, IsotonicCalibrator, MonitorPolicy,
                     EscalationGate, OutcomeLog)

policy, gate = MonitorPolicy(), EscalationGate()
fusion, calib = Fusion(), IsotonicCalibrator()      # refit both offline as logs grow
log = OutcomeLog("outcomes.jsonl")

def monitor(draft, domain, cost_error, internal_bet=False):
    tier = policy.choose(cost_error, internal_bet=internal_bet)

    s = Signals(verbal_conf=draft.stated_confidence)
    if "retrieval_support" in policy.TIERS[tier]:
        s.retrieval_support = entailment_share(draft)          # your retriever + NLI
    tier = policy.maybe_upgrade(tier, s)                        # cheap signals disagree?
    if "p_true" in policy.TIERS[tier]:
        s.p_true = p_true_pass(draft)                           # one self-eval call
    if "semantic_entropy" in policy.TIERS[tier]:
        s.semantic_entropy, s.n_samples = semantic_entropy(draft, n=10)

    raw = fusion.raw(s)
    p = float(calib(raw))
    decision = gate.decide(p, cost_error, surprise_z=prediction_error_z())
    return s, raw, p, decision
```

When the real outcome is known (from a tool result, a test, or the user — never from the agent grading itself), write it:

```python
log.write(domain, s, raw, p, escalated=decision["escalate"], correct=outcome)
```

## Offline refit

```python
from metacog import report
recs = log.load()
fusion = Fusion().fit(recs)                     # keeps the prior until n >= 100
calib  = IsotonicCalibrator().fit([r["raw"] for r in recs], [r["correct"] for r in recs])
capability_map = report(recs)                   # per-domain Brier, Murphy dx, ECE, AUROC2, gate recall
```

Adoption order: Murphy summary → domain tags → per-domain report at n ≥ 10 → fit fusion weights.

## API

| Object | Purpose | Key parameters |
|---|---|---|
| `brier(p, y)` | Brier score with reliability, resolution, uncertainty | `n_bins=10` |
| `murphy_summary(p, y)` | Brier skill score plus a plain-language diagnosis | — |
| `ece(p, y)` | Expected calibration error | `n_bins=15` |
| `auroc2(conf, correct)` | Type-2 AUROC (Mann–Whitney) | — |
| `fit_meta_d(stim, resp, rating, n_ratings)` | MLE meta-d′ and M-ratio | binary tasks only |
| `IsotonicCalibrator` | Raw score → calibrated p | `.fit(raw, y)`, call to apply |
| `Signals` | Container; any signal may be `None` | `verbal_conf`, `retrieval_support`, `p_true`, `semantic_entropy`, `n_samples` |
| `Fusion` | Weighted fusion, missing-signal renormalisation | `.fit(records, l2=5.0, min_n=100)` |
| `MonitorPolicy` | Which signals to compute, and their cost | `full_cost_error=20`, `mid_cost_error=5`, `disagree=0.35` |
| `EscalationGate` | System 1 → System 2 handoff | `cost_verify=1`, `verify_recall=0.8`, `surprise_override=2` |
| `OutcomeLog` | Append-only JSONL | `.write()`, `.load()` |
| `report(records)` | Per-domain capability map | ≥ 10 outcomes per domain |

## Tiers and costs

| Tier | Signals | Cost (units) | Used for |
|---|---|---|---|
| none | stated confidence | 0 | chit-chat, trivially reversible |
| cheap | + retrieval support | 1.5 | default per-turn |
| mid | + P(True) | 2.5 | cost of error ≥ 5 |
| full | + semantic entropy (N=10) | 12.5 | cost of error ≥ 20, internal bets |

Costs are placeholders in `TIER_COST`.
