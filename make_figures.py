"""Regenerate every figure in the repo root from metacog.py's own self-test data.
Run from the repo root:  python make_figures.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metacog import (fit_meta_d, IsotonicCalibrator, Fusion, Signals, EscalationGate,
                     MonitorPolicy, SIGNAL_NAMES, PRIOR_W, auroc2, ece)

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
INK, MUTED, A, B, C, D = "#1f2430", "#6b7280", "#4f46e5", "#f97316", "#10b981", "#e11d48"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#d1d5db",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 13, "axes.titlecolor": INK, "figure.dpi": 150})

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)

rng = np.random.default_rng(0)

# ---- 1. meta-d' : ideal vs noisy confidence (same draws as the self-test)
n, d_true, K = 6000, 1.5, 4
stim = rng.integers(0, 2, n)
x1 = rng.normal(np.where(stim == 1, d_true / 2, -d_true / 2), 1)
resp = (x1 > 0).astype(int)
fits = {}
for label, sd in (("Ideal observer", 0.0), ("Noisy confidence", 1.0)):
    x2 = x1 + rng.normal(0, 1, n) * sd
    strength = np.where(resp == 1, x2, -x2)
    rating = np.digitize(strength, [0.4, 0.9, 1.5]) + 1
    fits[label] = fit_meta_d(stim, resp, rating, K)

fig, ax = plt.subplots(figsize=(7, 4))
labels = list(fits)
xs = np.arange(len(labels)); w = 0.36
dp = [fits[l]["d_prime"] for l in labels]; md = [fits[l]["meta_d"] for l in labels]
b1 = ax.bar(xs - w / 2, dp, w, color="#c7d2fe", label="d′ (type-1 sensitivity)")
b2 = ax.bar(xs + w / 2, md, w, color=A, label="meta-d′ (type-2 sensitivity)")
for i, l in enumerate(labels):
    ax.text(xs[i], max(dp[i], md[i]) + 0.12, f"M-ratio {fits[l]['m_ratio']:.2f}", ha="center",
            fontweight="bold", color=INK)
ax.set_xticks(xs, labels); ax.set_ylim(0, 2.5); ax.set_ylabel("sensitivity")
ax.set_title("Metacognitive efficiency recovered by fit_meta_d()")
ax.legend(frameon=False, loc="upper right")
save(fig, "meta_d.png")

# ---- 2. reliability diagram: overconfident raw vs isotonic-calibrated
raw = rng.uniform(0.3, 1.0, 2000)
y = (rng.uniform(size=2000) < raw ** 2).astype(int)
cal = IsotonicCalibrator().fit(raw[:1000], y[:1000])
p_test, y_test, r_test = cal(raw[1000:]), y[1000:], raw[1000:]

def rel_curve(p, yy, nb=10):
    bins = np.clip((p * nb).astype(int), 0, nb - 1)
    pts = [(p[bins == b].mean(), yy[bins == b].mean(), (bins == b).sum()) for b in range(nb) if (bins == b).sum() >= 15]
    return np.array(pts)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], ls="--", color="#9ca3af", lw=1, label="perfect calibration")
for p, col, lab in ((r_test, B, f"raw score  (ECE {ece(r_test, y_test):.2f})"),
                    (p_test, A, f"isotonic   (ECE {ece(p_test, y_test):.2f})")):
    c = rel_curve(p, y_test)
    ax.plot(c[:, 0], c[:, 1], "-o", color=col, lw=2.2, ms=6, label=lab)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xlabel("predicted reliability p"); ax.set_ylabel("observed accuracy")
ax.set_title("Reliability diagram: recalibrating an overconfident monitor")
ax.legend(frameon=False, loc="upper left")
save(fig, "reliability.png")

# ---- 3. fusion weights: asserted prior vs fitted on logs
recs = []
for _ in range(1500):
    truth = rng.uniform() < 0.6
    se = max(0.0, rng.normal(0.8 if truth else 1.0, 0.5))
    recs.append({"verbal_conf": float(np.clip(rng.normal(0.85, 0.1), 0, 1)),
                 "retrieval_support": float(np.clip(rng.normal(0.7 if truth else 0.4, 0.2), 0, 1)),
                 "p_true": float(np.clip(rng.normal(0.7 if truth else 0.6, 0.15), 0, 1)),
                 "semantic_entropy": se, "n_samples": 10, "correct": int(truth)})
fz = Fusion().fit(recs[:1000])
def au(f):
    r = [f.raw(Signals(**{k: x[k] for k in Signals.__dataclass_fields__})) for x in recs[1000:]]
    return auroc2(r, [x["correct"] for x in recs[1000:]])
prior_rel = PRIOR_W / PRIOR_W.sum(); fit_rel = fz.w / fz.w.sum()
nice = {"agree": "semantic agreement", "p_true": "P(True)", "verbal_conf": "stated confidence",
        "retrieval_support": "retrieval support"}
fig, ax = plt.subplots(figsize=(7.5, 4))
ys = np.arange(len(SIGNAL_NAMES)); h = 0.38
ax.barh(ys + h / 2, prior_rel, h, color="#d1d5db", label=f"asserted prior  (AUROC₂ {au(Fusion()):.2f})")
ax.barh(ys - h / 2, fit_rel, h, color=C, label=f"fitted on 1,000 logs  (AUROC₂ {au(fz):.2f})")
ax.set_yticks(ys, [nice[s] for s in SIGNAL_NAMES]); ax.invert_yaxis()
ax.set_xlabel("relative weight"); ax.set_xlim(0, 0.6)
ax.set_title("Fusion.fit() learns which signals actually carry information")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=2)
save(fig, "fusion_weights.png")

# ---- 4. escalation threshold vs stakes
gate = EscalationGate()
ce = np.linspace(1.3, 30, 300)
fig, ax = plt.subplots(figsize=(7.5, 4))
for rec, col in ((0.5, "#a5b4fc"), (0.8, A), (0.95, "#312e81")):
    g = EscalationGate(verify_recall=rec)
    ax.plot(ce, [g.threshold(c) for c in ce], color=col, lw=2.2, label=f"verify recall {rec:.0%}")
pol = MonitorPolicy()
for c, lab in ((1, "chat reply"), (6, "calendar edit"), (20, "irreversible action")):
    if c > 1.3:
        t = gate.threshold(c)
        ax.scatter([c], [t], color=B, zorder=5, s=40)
        ax.annotate(f"{lab}\nτ = {t:.2f}, tier: {pol.choose(c)}", (c, t), xytext=(8, -32),
                    textcoords="offset points", fontsize=9.5, color=INK)
ax.fill_between(ce, 0, [gate.threshold(c) for c in ce], color=A, alpha=0.06)
ax.text(24, 0.35, "escalate to System 2\n(p below τ)", ha="center", color=A, fontsize=10)
ax.set_ylim(0, 1.02); ax.set_xlabel("cost of an error (× cost of verifying)")
ax.set_ylabel("threshold τ on calibrated p")
ax.set_title("EscalationGate: when acting now costs more than checking")
ax.legend(frameon=False, loc="lower right")
save(fig, "escalation_gate.png")

# ---- 5. architecture: one cognitive cycle
fig, ax = plt.subplots(figsize=(12, 7.2)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 7.3)
def box(x, y, w, h, title, sub, col, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fill, ec=col, lw=1.6))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.62, sub, ha="center", va="top", fontsize=8.6, color=MUTED, linespacing=1.3)
def arrow(a, b, col=MUTED, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, color=col, lw=1.4,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))
box(0.2, 4.4, 2.1, 1.6, "1  Intake", "percepts with source,\ntime, salience prior", "#9ca3af", "#f9fafb")
box(2.8, 4.4, 2.6, 1.6, "2  Specialists", "System 1, in parallel:\nmemory, planner, valence,\nnorms, tools", "#9ca3af", "#f9fafb")
box(5.9, 4.4, 2.6, 1.6, "3–4  Global workspace", "salience competition,\nbroadcast + ignition", A, "#eef2ff")
box(9.0, 4.4, 2.8, 1.6, "5  Fast proposal", "draft answer or action\n+ stated confidence", "#9ca3af", "#f9fafb")
box(8.6, 1.9, 3.2, 1.9, "6  Metacognitive monitor", "MonitorPolicy → Signals\nFusion → IsotonicCalibrator\n= calibrated p", B, "#fff7ed")
box(4.8, 1.9, 3.2, 1.9, "7  Escalation gate", "p < τ(stakes)?\nyes → System 2 with\nexternal grounding", B, "#fff7ed")
box(1.4, 1.9, 2.8, 1.9, "8–9  Act + predict", "approval gate, predict\noutcome, compare after", "#9ca3af", "#f9fafb")
box(1.4, 0.15, 10.4, 1.25, "10  OutcomeLog → report() → persistent self-model",
    "per-domain Brier, Murphy diagnosis, ECE, AUROC₂, meta-d′ · refit Fusion and calibrator offline", C, "#ecfdf5")
arrow((2.3, 5.2), (2.8, 5.2)); arrow((5.4, 5.2), (5.9, 5.2), A); arrow((8.5, 5.2), (9.0, 5.2))
arrow((10.4, 4.4), (10.2, 3.8), B); arrow((8.6, 2.85), (8.0, 2.85), B); arrow((4.8, 2.85), (4.2, 2.85))
arrow((2.8, 1.9), (2.8, 1.4), C); arrow((10.2, 1.4), (10.2, 1.9), C, ls="--")
arrow((7.2, 6.0), (4.1, 6.0), A, rad=0.35)
ax.text(5.65, 6.62, "broadcast to every specialist next tick", ha="center", fontsize=8.6, color=A)
ax.text(11.95, 1.62, "refit", ha="right", fontsize=8.6, color=C)
ax.text(0.2, 7.25, "One cognitive cycle", fontsize=14, fontweight="bold", color=INK, va="top")
ax.text(0.2, 6.9, "orange = implemented in metacog.py", fontsize=9, color=B, va="top")
save(fig, "architecture.png")
print("wrote", sorted(os.listdir(OUT)))
