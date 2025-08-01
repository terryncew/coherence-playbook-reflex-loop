# csl.py
"""
Constraint Surfing Layer (CSL)
- Filters reflex signals through contextual constraints before nudging.
- Adds damping (hysteresis), cooldowns, and deterministic ordering.
- Integrates Open-Line signals (phi_star, kappa, epsilon, vkd).

No external deps. Python 3.10+.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Any
import time


# --------------------------- Enums ---------------------------

class ConstraintType(Enum):
    ENERGY = "energy"
    SOCIAL = "social"
    TEMPORAL = "temporal"
    COGNITIVE = "cognitive"
    NARRATIVE = "narrative"
    SYSTEM = "system"
    SAFETY = "safety"


class ModulationStrategy(Enum):
    AMPLIFY = "amplify"
    DAMPEN = "dampen"
    REDIRECT = "redirect"
    DEFER = "defer"
    MIRROR = "mirror"
    REFRAME = "reframe"


# --------------------------- Context ---------------------------

@dataclass
class ContextState:
    """Current runtime context (0..1 scales unless noted)."""
    energy_level: float = 1.0
    social_tension: float = 0.0
    memory_load: float = 0.0
    urgency: float = 0.0
    coherence_drift: float = 0.0
    audience_sensitivity: float = 0.5
    # Open-Line signals (optional but supported)
    phi_star: Optional[float] = None
    kappa: Optional[float] = None
    epsilon: Optional[float] = None
    vkd: Optional[float] = None  # >=0 viable; <0 unsafe
    # bookkeeping
    timestamp: float = field(default_factory=time.time)


# --------------------------- Constraint ---------------------------

Evaluator = Callable[[ContextState], float]               # -> strength in [0,1]
Modulator = Callable[[Dict[str, Any], float], Dict[str, Any]]  # modifies signal


@dataclass
class Constraint:
    name: str
    ctype: ConstraintType
    priority: int                              # lower is applied first
    threshold_on: float                        # activate when strength >= on
    threshold_off: float                       # deactivate when strength < off  (<= on for hysteresis)
    cooldown_s: float                          # min seconds between activations
    strategy: ModulationStrategy
    evaluator: Evaluator
    modulator: Modulator

    # runtime state (per-instance)
    _active: bool = field(default=False, init=False)
    _last_change_ts: float = field(default=0.0, init=False)

    def should_apply(self, ctx: ContextState) -> Tuple[bool, float]:
        """Hysteresis + cooldown gate."""
        strength = max(0.0, min(1.0, self.evaluator(ctx)))
        now = time.time()

        if self._active:
            # Stay active until we drop below threshold_off
            if strength < self.threshold_off:
                self._active = False
                self._last_change_ts = now
        else:
            # Only activate if above threshold_on AND cooldown passed
            if strength >= self.threshold_on and (now - self._last_change_ts) >= self.cooldown_s:
                self._active = True
                self._last_change_ts = now

        return self._active, strength


# --------------------------- CSL Core ---------------------------

class ConstraintSurfingLayer:
    def __init__(self):
        self._domains: Dict[str, List[Constraint]] = {}
        self._ctx: ContextState = ContextState()
        self._history: List[ContextState] = []
        self._max_hist = 12
        self._install_defaults()

    # ---- public API ----
    def update_context(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self._ctx, k):
                setattr(self._ctx, k, v)
        self._ctx.timestamp = time.time()
        self._history.append(self._ctx)
        if len(self._history) > self._max_hist:
            self._history.pop(0)

    def add_constraint(self, domain: str, constraint: Constraint) -> None:
        self._domains.setdefault(domain, []).append(constraint)
        # keep deterministic order by priority then name
        self._domains[domain].sort(key=lambda c: (c.priority, c.name))

    def remove_constraint(self, domain: str, name: str) -> None:
        if domain in self._domains:
            self._domains[domain] = [c for c in self._domains[domain] if c.name != name]

    def modulate_nudge(self, domain: str, reflex_signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply active constraints (ordered) to a reflex signal.
        Returns a new dict with audit trail.
        """
        # base signal defaults
        signal = {
            "intensity": 1.0,     # 0..∞  (we’ll clamp)
            "urgency": 0.5,       # 0..1
            "directness": 1.0,    # 0..1
            "tone": "neutral",
            "strategy": "direct",
            "note": "",
            **reflex_signal,
        }

        # Global safety: VKD < 0 means exit clever mode
        audit: List[Dict[str, Any]] = []
        if self._ctx.vkd is not None and self._ctx.vkd < 0:
            signal["strategy"] = "safety_minimize"
            signal["intensity"] = min(signal["intensity"], 0.4)
            signal["urgency"] = max(signal["urgency"], 0.8)  # urgent to minimize harm
            signal["note"] += " | VKD<0: safety mode"
            audit.append({"constraint": "vkd_safety", "strength": 1.0, "effect": "safety_minimize"})

        # Domain + default constraints
        constraints = list(self._domains.get("default", [])) + list(self._domains.get(domain, []))

        # Evaluate & apply
        for c in constraints:
            active, strength = c.should_apply(self._ctx)
            if not active:
                continue
            before = dict(signal)
            signal = c.modulator(signal, strength)
            audit.append({
                "constraint": c.name,
                "ctype": c.ctype.value,
                "strategy": c.strategy.value,
                "strength": round(strength, 3),
                "delta": _diff(before, signal),
            })

        # Clamp and finalize
        signal["intensity"] = max(0.0, signal["intensity"])
        signal["urgency"] = max(0.0, min(1.0, signal["urgency"]))
        signal["directness"] = max(0.0, min(1.0, signal["directness"]))
        signal["audit"] = audit
        signal["active_constraints"] = [a["constraint"] for a in audit]
        return signal

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "context": self._ctx,
            "domains": list(self._domains.keys()),
            "constraints": {d: [c.name for c in cs] for d, cs in self._domains.items()},
            "history_len": len(self._history),
        }

    # ---- defaults ----
    def _install_defaults(self) -> None:
        # Energy: dampen when low
        self.add_constraint("default", Constraint(
            name="low_energy_dampen",
            ctype=ConstraintType.ENERGY,
            priority=10,
            threshold_on=0.7,             # active when (1 - energy) >= 0.7  => energy <= 0.3
            threshold_off=0.5,
            cooldown_s=2.0,
            strategy=ModulationStrategy.DAMPEN,
            evaluator=lambda ctx: max(0.0, 1.0 - ctx.energy_level),
            modulator=_mod_energy,
        ))

        # Social: mirror/soften when tension * sensitivity high
        self.add_constraint("default", Constraint(
            name="social_mirror",
            ctype=ConstraintType.SOCIAL,
            priority=20,
            threshold_on=0.6,
            threshold_off=0.4,
            cooldown_s=1.0,
            strategy=ModulationStrategy.MIRROR,
            evaluator=lambda ctx: max(0.0, min(1.0, ctx.social_tension * ctx.audience_sensitivity)),
            modulator=_mod_social,
        ))

        # Cognitive: defer or chunk when memory load high
        self.add_constraint("default", Constraint(
            name="cognitive_defer",
            ctype=ConstraintType.COGNITIVE,
            priority=30,
            threshold_on=0.7,
            threshold_off=0.5,
            cooldown_s=3.0,
            strategy=ModulationStrategy.DEFER,
            evaluator=lambda ctx: ctx.memory_load,
            modulator=_mod_cognitive,
        ))

        # Narrative drift: reframe when drift rises
        self.add_constraint("default", Constraint(
            name="drift_reframe",
            ctype=ConstraintType.NARRATIVE,
            priority=40,
            threshold_on=0.5,
            threshold_off=0.35,
            cooldown_s=1.5,
            strategy=ModulationStrategy.REFRAME,
            evaluator=lambda ctx: ctx.coherence_drift if ctx.coherence_drift is not None else 0.0,
            modulator=_mod_drift,
        ))

        # Phi* low: tighten plan when coherence-per-cost dips (if present)
        self.add_constraint("default", Constraint(
            name="phi_star_guard",
            ctype=ConstraintType.SYSTEM,
            priority=50,
            threshold_on=0.6,  # strength = (1 - phi*) -> active when phi* <= 0.4
            threshold_off=0.45,
            cooldown_s=2.0,
            strategy=ModulationStrategy.REDIRECT,
            evaluator=lambda ctx: (1.0 - ctx.phi_star) if (ctx.phi_star is not None) else 0.0,
            modulator=_mod_phi_star,
        ))


# --------------------------- Modulators ---------------------------

def _append(note: str, extra: str) -> str:
    return (note + " | " + extra).strip(" |")

def _mod_energy(sig: Dict[str, Any], strength: float) -> Dict[str, Any]:
    # Stronger low-energy => more dampening and deferral
    damp = 0.5 + 0.5 * strength  # 0.5..1.0
    sig["intensity"] *= max(0.1, 1.0 - 0.8 * strength)
    sig["urgency"] *= max(0.2, 1.0 - 0.6 * strength)
    if strength > 0.8:
        sig["strategy"] = "defer_and_simplify"
    sig["note"] = _append(sig["note"], f"energy↓ s={strength:.2f}")
    return sig

def _mod_social(sig: Dict[str, Any], strength: float) -> Dict[str, Any]:
    if strength > 0.75:
        sig["tone"] = "diplomatic"
        sig["directness"] *= 0.6
        sig["strategy"] = "mirror_and_validate"
    else:
        sig["tone"] = "gentle"
        sig["directness"] *= 0.85
    sig["note"] = _append(sig["note"], f"social_tension s={strength:.2f}")
    return sig

def _mod_cognitive(sig: Dict[str, Any], strength: float) -> Dict[str, Any]:
    if strength > 0.85:
        sig["strategy"] = "defer"
        sig["complexity"] = "minimal"
    else:
        sig["complexity"] = "reduced"
        sig["chunking"] = True
    sig["note"] = _append(sig["note"], f"cog_load s={strength:.2f}")
    return sig

def _mod_drift(sig: Dict[str, Any], strength: float) -> Dict[str, Any]:
    if strength > 0.7:
        sig["strategy"] = "reframe_context"
        sig["priority"] = "coherence_restoration"
    else:
        sig["strategy"] = "clarify_intent"
    sig["note"] = _append(sig["note"], f"drift s={strength:.2f}")
    return sig

def _mod_phi_star(sig: Dict[str, Any], strength: float) -> Dict[str, Any]:
    # strength = 1 - phi_star, so higher strength => lower phi*
    sig["plan_depth"] = max(1, int(sig.get("plan_depth", 3) - max(1, round(2 * strength))))
    sig["tools_fanout"] = max(0, int(sig.get("tools_fanout", 2) - max(1, round(2 * strength))))
    sig["note"] = _append(sig["note"], f"phi* guard s={strength:.2f}")
    sig["strategy"] = "shorten_and_refresh"
    return sig


def _diff(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    delta = {}
    for k, v in after.items():
        if k not in before or before[k] != v:
            delta[k] = v
    return delta


# --------------------------- Example ---------------------------

if __name__ == "__main__":
    csl = ConstraintSurfingLayer()
    print("== Normal context ==")
    csl.update_context(energy_level=0.8, social_tension=0.2, coherence_drift=0.1, phi_star=0.7, vkd=0.2)
    sig = {"type": "coherence_nudge", "message": "Clarify intent", "intensity": 0.7, "plan_depth": 4, "tools_fanout": 2}
    print(csl.modulate_nudge("conversation", sig))

    print("\n== High social tension ==")
    csl.update_context(social_tension=0.9, audience_sensitivity=0.9)
    print(csl.modulate_nudge("conversation", sig))

    print("\n== Low energy + high load + drift ==")
    csl.update_context(energy_level=0.2, memory_load=0.9, coherence_drift=0.8)
    print(csl.modulate_nudge("conversation", sig))

    print("\n== VKD < 0 safety mode ==")
    csl.update_context(vkd=-0.1)
    print(csl.modulate_nudge("conversation", sig))

    print("\n== Diagnostics ==")
    print(csl.diagnostics())
