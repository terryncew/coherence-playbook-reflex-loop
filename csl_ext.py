# csl_ext.py
"""
CSL Extensions for ConstraintSurfingLayer
- Learning (EWMA effectiveness)
- Adaptive thresholds (bounded, rate-limited)
- Composite constraints
- Context pattern hints
- Conflict arbitration (priority → strength → safety)
- Simple profiler

Designed to wrap an existing `ConstraintSurfingLayer` (from csl.py).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Tuple
import time
import math
import functools
import threading


# ------------------------- Learning / Metrics -------------------------

@dataclass
class ConstraintMetrics:
    activations: int = 0
    ewma_strength: float = 0.0
    ewma_outcome: float = 0.5  # neutral prior
    alpha_strength: float = 0.2
    alpha_outcome: float = 0.2
    last_ts: float = field(default_factory=time.time)

    def record(self, strength: float, outcome: Optional[float] = None) -> None:
        self.activations += 1
        self.ewma_strength = (1.0 - self.alpha_strength) * self.ewma_strength + self.alpha_strength * strength
        if outcome is not None:
            self.ewma_outcome = (1.0 - self.alpha_outcome) * self.ewma_outcome + self.alpha_outcome * outcome
        self.last_ts = time.time()

    @property
    def effectiveness(self) -> float:
        return self.ewma_outcome  # 0..1


# ------------------------- Adaptive Thresholds -------------------------

class AdaptiveConstraint:
    """
    Wraps a base constraint (from CSL) and nudges its thresholds using effectiveness.
    Ensures bounds and rate limits to avoid flapping.
    """
    def __init__(
        self,
        base_constraint,
        metrics: ConstraintMetrics | None = None,
        adapt_rate: float = 0.03,    # small nudges
        min_on: float = 0.15,
        max_on: float = 0.9,
        min_off: float = 0.05,
        max_off: float = 0.85,
        cool_s: float = 10.0,        # min seconds between adjustments
    ):
        self.base = base_constraint
        self.metrics = metrics or ConstraintMetrics()
        self.adapt_rate = adapt_rate
        self.min_on, self.max_on = min_on, max_on
        self.min_off, self.max_off = min_off, max_off
        self.cool_s = cool_s
        self._last_adapt_ts = 0.0
        self.history: List[Dict[str, float]] = []

    def maybe_adapt(self) -> None:
        now = time.time()
        if now - self._last_adapt_ts < self.cool_s or self.metrics.activations < 5:
            return

        eff = self.metrics.effectiveness  # 0..1
        # push on/off down if effective (>0.7); up if ineffective (<0.3)
        if eff > 0.7:
            delta = self.adapt_rate * (eff - 0.5)
            self.base.threshold_on = max(self.min_on, self.base.threshold_on - delta)
            self.base.threshold_off = max(self.min_off, self.base.threshold_off - delta)
        elif eff < 0.3:
            delta = self.adapt_rate * (0.5 - eff)
            self.base.threshold_on = min(self.max_on, self.base.threshold_on + delta)
            self.base.threshold_off = min(self.max_off, self.base.threshold_off + delta)

        self._last_adapt_ts = now
        self.history.append({
            "ts": now,
            "eff": eff,
            "on": self.base.threshold_on,
            "off": self.base.threshold_off,
        })


# ------------------------- Composite Constraints -------------------------

class CompositeConstraint:
    """
    Combines multiple constraints by AND/OR/WEIGHTED modes for a derived evaluator.
    Useful to gate expensive modulators behind multiple signals.
    """
    def __init__(self, name: str, constraints: List[Any], mode: str = "AND"):
        self.name = name
        self.constraints = constraints
        self.mode = mode.upper()

    def evaluator(self, ctx) -> float:
        strengths = []
        for c in self.constraints:
            if hasattr(c, "evaluator"):
                strengths.append(float(c.evaluator(ctx)))
            elif hasattr(c, "evaluate"):
                strengths.append(float(c.evaluate(ctx)))
        if not strengths:
            return 0.0
        if self.mode == "AND":
            return min(strengths)
        if self.mode == "OR":
            return max(strengths)
        # WEIGHTED_AVG: equal weights by default
        return sum(strengths) / len(strengths)


# ------------------------- Pattern Detector -------------------------

class ContextPatternDetector:
    """
    Buckets recent context into a coarse signature; learns which constraints tend to activate.
    This is a *hint* system, not a hard gate.
    """
    def __init__(self, window: int = 3, min_freq: int = 3):
        self.window = window
        self.min_freq = min_freq
        self._patterns: Dict[str, Dict[str, int]] = {}   # sig -> constraint_name -> count
        self._freq: Dict[str, int] = {}                  # sig -> frequency

    def _bucket(self, x: Optional[float], bins: int = 4) -> int:
        if x is None: return 0
        return max(0, min(bins, int(x * bins)))

    def signature(self, hist: List[Any]) -> Optional[str]:
        if len(hist) < self.window:
            return None
        recent = hist[-self.window:]
        sig = []
        for ctx in recent:
            sig.append((
                self._bucket(getattr(ctx, "energy_level", None)),
                self._bucket(getattr(ctx, "social_tension", None)),
                self._bucket(getattr(ctx, "memory_load", None)),
                self._bucket(getattr(ctx, "coherence_drift", None)),
                self._bucket(getattr(ctx, "phi_star", None)),
            ))
        return str(hash(tuple(sig)))

    def record(self, sig: Optional[str], active_constraints: List[str]) -> None:
        if sig is None: return
        self._freq[sig] = self._freq.get(sig, 0) + 1
        if sig not in self._patterns:
            self._patterns[sig] = {}
        counts = self._patterns[sig]
        for name in active_constraints:
            counts[name] = counts.get(name, 0) + 1

    def predict(self, sig: Optional[str]) -> List[str]:
        if sig is None or sig not in self._patterns:
            return []
        freq = self._freq.get(sig, 0)
        if freq < self.min_freq:
            return []
        counts = self._patterns[sig]
        return [k for k, v in counts.items() if v >= 0.5 * freq]


# ------------------------- Conflict Arbitration -------------------------

class ConstraintArbitrator:
    """
    Resolve contradictory strategies with a simple, explainable policy:
    1) Higher safety priority (e.g., VKD) wins.
    2) Lower numeric priority (from base constraint) wins.
    3) Stronger strength wins.
    """
    SAFETY_NAMES = {"vkd_safety"}

    def choose(self, active: List[Tuple[Any, float]]) -> List[Any]:
        if not active:
            return []
        # If any safety constraints, keep them and drop direct conflicts
        safes = [c for (c, s) in active if getattr(c, "name", "") in self.SAFETY_NAMES]
        if safes:
            return [c for (c, s) in active if c in safes]

        # Otherwise, sort by (priority asc, strength desc, name)
        ranked = sorted(active, key=lambda cs: (getattr(cs[0], "priority", 999), -cs[1], getattr(cs[0], "name", "")))
        # Keep all that do not directly contradict the top choice's strategy
        top = ranked[0][0]
        top_strategy = getattr(top, "strategy", None)
        winners = [top]
        for c, s in ranked[1:]:
            st = getattr(c, "strategy", None)
            if not self._conflicts(top_strategy, st):
                winners.append(c)
        return winners

    def _conflicts(self, s1, s2) -> bool:
        # Minimal conflict table; extend as needed
        pairs = {("AMPLIFY", "DAMPEN"), ("DEFER", "AMPLIFY")}
        v1 = getattr(s1, "value", str(s1)) if s1 else ""
        v2 = getattr(s2, "value", str(s2)) if s2 else ""
        return (v1, v2) in pairs or (v2, v1) in pairs


# ------------------------- Profiler -------------------------

class CSLProfiler:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_eval = 0
        self.total_mod = 0
        self.sum_eval_t = 0.0
        self.sum_mod_t = 0.0
        self.hotspots: Dict[str, Dict[str, float]] = {}

    def timeit(self, kind: str, name: str, fn: Callable, *args, **kwargs):
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        dur = time.perf_counter() - start
        with self.lock:
            if kind == "eval":
                self.total_eval += 1
                self.sum_eval_t += dur
            else:
                self.total_mod += 1
                self.sum_mod_t += dur
            hs = self.hotspots.setdefault(name, {"count": 0, "time": 0.0})
            hs["count"] += 1
            hs["time"] += dur
        return out

    def report(self) -> Dict[str, Any]:
        with self.lock:
            rep = {
                "avg_eval_ms": (self.sum_eval_t / max(1, self.total_eval)) * 1e3,
                "avg_mod_ms": (self.sum_mod_t / max(1, self.total_mod)) * 1e3,
                "hotspots": {
                    k: {"avg_ms": v["time"] / max(1, v["count"]) * 1e3, "count": v["count"]}
                    for k, v in self.hotspots.items()
                },
            }
        return rep


# ------------------------- Extended Wrapper -------------------------

class CSLX:
    """
    Wraps an existing ConstraintSurfingLayer (CSL) and adds:
    - pattern hints,
    - conflict arbitration,
    - adaptive thresholds with metrics,
    - profiling and outcome feedback.
    """
    def __init__(self, base_csl):
        self.csl = base_csl
        self.patterns = ContextPatternDetector()
        self.arbitrator = ConstraintArbitrator()
        self.profiler = CSLProfiler()
        # map: constraint_name -> AdaptiveConstraint
        self.adaptives: Dict[str, AdaptiveConstraint] = {}

    def register_adaptive(self, domain: str, name: str, wrapper: AdaptiveConstraint) -> None:
        # caller constructs wrapper with the base constraint
        self.adaptives[name] = wrapper

    def modulate_nudge(self, domain: str, reflex_signal: Dict[str, Any]) -> Dict[str, Any]:
        # 1) Predict likely constraints from recent pattern (hint only)
        sig = self.patterns.signature(getattr(self.csl, "_history", []))
        _ = self.patterns.predict(sig)  # available for UI/debug if you want

        # 2) Run base CSL with profiling
        result = self.profiler.timeit("mod", "modulate_nudge", self.csl.modulate_nudge, domain, reflex_signal)

        # 3) Conflict arbitration (if you want to reduce to a non-contradictory subset)
        #    We need constraint objects & strengths; your CSL returns names, so skip or extend CSL to expose (obj, strength).
        #    For now we leave result as-is, but record the pattern for learning:
        actives = result.get("active_constraints", [])
        self.patterns.record(sig, actives)

        # 4) Update adaptive metrics; opportunistically adapt thresholds
        for name in actives:
            a = self.adaptives.get(name)
            if a:
                a.metrics.record(strength=1.0)  # if you have true strength, pass it here
                a.maybe_adapt()

        return result

    def record_outcome(self, constraint_name: str, outcome_score: float) -> None:
        """Call this after you see the real-world effect of a nudge (0..1)."""
        a = self.adaptives.get(constraint_name)
        if a:
            a.metrics.record(strength=1.0, outcome=outcome_score)
            a.maybe_adapt()

    def profiler_report(self) -> Dict[str, Any]:
        return self.profiler.report()
