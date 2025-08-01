# app.py
from csl import ConstraintSurfingLayer
from mel import MELEnhancedReflex, EventType, Severity

csl = ConstraintSurfingLayer()
mel = MELEnhancedReflex()

# 1) Log warnings/critical/collapse when your Open-Line dials cross thresholds
def on_tick(open_line):
    # Example thresholds — tune to your stack
    if open_line["phi_star"] is not None and open_line["phi_star"] < 0.45:
        mel.log(EventType.PHI_DEGRADATION, Severity.WARNING, 1 - open_line["phi_star"])

    if open_line["vkd"] is not None and open_line["vkd"] < 0:
        mel.log(EventType.VKD_CRITICAL, Severity.CRITICAL, -open_line["vkd"])

    # 2) Ask MEL if we should pre-emptively nudge
    should, conf, why = mel.should_pre_nudge(EventType.PHI_DEGRADATION)
    reflex = {"type": "coherence_nudge", "message": "shorten/refresh", "intensity": 0.8}
    if should:
        reflex["note"] = f"pre_nudge({conf:.2f}): {why}"
        reflex["strategy"] = "shorten_and_refresh"

    # 3) Pass the reflex into CSL for modulation
    out = csl.modulate_nudge("conversation", reflex)
    return out
