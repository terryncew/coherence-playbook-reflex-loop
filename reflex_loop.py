loop:
  x_t = observe_inputs()
  y_t = current_outputs()

  phi = compute_phi_star(state_t, x_t, y_t, cost_t)
  vkd = estimate_viability_margin(state_t, dynamics, lag)

  if vkd < 0:
    enter_damage_minimization()
    continue

  if phi < PHI_SOFT or falling_fast(phi):
    shorten_horizon()
    slow_intent_updates()
    refresh_stale_context()

  if drains_phi_star(thread_i):
    apoptose(thread_i); reallocate_budget()

  mode = allostatic_mode(phi_trend, phi_volatility, hysteresis)
  set_planning_bounds(mode)

  verify(phi_after, vkd_after); log_intervention_effects()
