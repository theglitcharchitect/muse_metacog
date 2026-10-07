# muse_metacog

**A calibrated metacognitive monitor for LLM agents: it scores how much to trust each claim, fixes overconfidence, learns which signals matter, and decides when to stop and verify.**

Built for the Muse agent and usable in any agent loop. `metacog.py` turns cheap and expensive uncertainty signals (stated confidence, retrieval support, P(True), semantic entropy) into one calibrated probability, gates System 1 → System 2 escalation on the cost of being wrong, and logs every outcome so the agent keeps an honest, per-domain map of what it is good at.

![Where the monitor sits in an agent loop](docs/img/architecture.png)

---

## What's in here

| Path | What it is |
|---|---|
| [`metacog.py`](metacog.py) | Metacognitive monitor v2: scoring, calibration, signal fusion, monitor cost policy, escalation gate, outcome log, per-domain report |
| [`docs/architecture.md`](docs/architecture.md) | The wider agent loop the monitor plugs into: 10-step cycle, self-model, evaluation harness, build order, risks |
| [`docs/literature.md`](docs/literature.md) | Background: 15-paper matrix on LLM metacognition, calibration, and agent architectures, every entry checked against its source |
| [`docs/metacog-guide.md`](docs/metacog-guide.md) | How to wire `metacog.py` into an agent loop, with the API for every class |
| [`docs/make_figures.py`](docs/make_figures.py) | Regenerates every figure below from the module's own self-test |

## Quick start

```bash
pip install -r requirements.txt
python metacog.py            # self-test
python docs/make_figures.py  # rebuild docs/img/*
```

Expected self-test output (seed 0):

```
            ideal: d'=1.50 meta-d'=1.54 M-ratio=1.03
 noisy-confidence: d'=1.50 meta-d'=0.78 M-ratio=0.52
raw  : brier 0.2172  reliability 0.0416  ECE 0.1888
calib: brier 0.1760  reliability 0.0017  ECE 0.0394  AUROC2 0.8132
murphy: miscalibration -> healthy
fusion: relative_weights agree 0.269, p_true 0.212, verbal_conf 0.0, retrieval_support 0.518
   prior AUROC2=0.838   fitted AUROC2=0.886
```

---

## The monitor, in four pictures

### 1. It recovers metacognitive efficiency
`fit_meta_d()` is a maximum-likelihood meta-d′ fit (Maniscalco & Lau 2012). An ideal observer scores M-ratio ≈ 1; adding noise to confidence alone halves it while type-1 accuracy stays fixed — exactly the dissociation the metric exists to catch.

![meta-d'](docs/img/meta_d.png)

### 2. It fixes overconfidence
Raw confidence scores are rarely calibrated. `IsotonicCalibrator` (pool-adjacent-violators) maps them onto observed accuracy, cutting ECE from 0.19 to 0.04 on held-out data. `murphy_summary()` diagnoses *why* the Brier score is what it is — here, "miscalibration" before and "healthy" after.

![Reliability diagram](docs/img/reliability.png)

### 3. It learns which signals to trust
The prior weights are an assertion (semantic entropy beats self-report, per Farquhar et al. 2024). `Fusion.fit()` runs an L2-to-prior logistic regression on logged outcomes, so small samples stay near the prior and large ones let the data speak. On synthetic logs where stated confidence is uninformative and retrieval support is strong, it drops the first to zero and doubles the second, lifting AUROC₂ from 0.84 to 0.89.

![Fusion weights](docs/img/fusion_weights.png)

### 4. It knows when to stop and check
`EscalationGate` hands off to System 2 when the expected cost of acting exceeds the cost of verifying: escalate if `p < 1 − cost_verify / (verify_recall × cost_error)`. `MonitorPolicy` picks how much monitoring a claim deserves *before* paying for it (none → cheap → mid → full).

![Escalation gate](docs/img/escalation_gate.png)

> System 2 must use **external grounding** — tools, tests, retrieval, the user. Intrinsic self-critique does not reliably fix errors (Huang et al., ICLR 2024).

---

## Metrics the self-model tracks

| Metric | Measures | Function |
|---|---|---|
| Brier + Murphy decomposition | Calibration and discrimination together (proper scoring rule) | `brier()`, `murphy_summary()` |
| ECE (15 bins) | Calibration gap, reported beside Brier | `ece()` |
| AUROC₂ | Does confidence separate right from wrong? | `auroc2()` |
| meta-d′, M-ratio | Metacognitive efficiency, binary tasks only | `fit_meta_d()` |
| Gate error recall, escalation rate | Does monitoring actually control behaviour? | `report()` |

`report()` turns the outcome stream into a per-domain capability map (n ≥ 10 per domain; meta-d′ at ≥ 50 binary trials).

## Roadmap

1. ~~Metacognitive monitor and calibration harness~~ (v2)
2. Domain tags on every outcome → per-domain report at n ≥ 10 → fit fusion weights on real logs
3. Memory tiers and outcome logging in Muse
4. Workspace competition
5. System 2 escalation with external verification
6. Prediction-error loop and self-model consolidation

## Caveats

- The figures use synthetic data from the self-test, not production logs.
- `TIER_COST` values are placeholders; tune them to your inference stack.
- `correct` labels must come from external ground truth, never from the agent grading itself.

## License

Apache License 2.0. See [LICENSE](LICENSE).
