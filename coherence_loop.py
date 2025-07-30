“””
GlycoShell: Boundary-Aware Intelligence Wrapper
MVP Implementation - Week 1 & 2 Sprint

A lightweight shell that wraps any LLM/agent with boundary intelligence.
Provides real-time coherence monitoring, viability detection, and adaptive processing modes.
“””

import numpy as np
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import threading
from abc import ABC, abstractmethod

# ============================================================================

# Core Types and Enums

# ============================================================================

class ProcessingMode(Enum):
EXPLORATION = “exploration”      # High Φ*, low threat
CONSERVATION = “conservation”    # Moderate Φ*, elevated threat  
REFLEX = “reflex”               # Low Φ* or high threat
DAMAGE_CONTROL = “damage_control” # VKD < 0

@dataclass
class SurfaceFeatures:
“”“Surface-level patterns detected before semantic processing”””
rhythm_score: float = 0.0        # Token spacing regularity
entropy_score: float = 0.0       # Information density
structure_score: float = 0.0     # Grammar/format coherence
valence_score: float = 0.0       # Emotional stress indicators
format_integrity: float = 0.0    # Template/schema compliance
timestamp: float = 0.0

@dataclass
class BoundarySignals:
“”“Real-time boundary intelligence metrics”””
phi_star: float = 0.0            # Coherence per unit cost
phi_slope: float = 0.0           # Coherence trend
phi_volatility: float = 0.0      # Coherence stability
vkd: float = 1.0                 # Viability kernel distance
mode: ProcessingMode = ProcessingMode.EXPLORATION
cost: float = 1.0                # Processing cost
timestamp: float = 0.0

@dataclass
class ProcessingNudge:
“”“Adaptive processing constraints based on boundary state”””
max_layers: Optional[int] = None
temperature: Optional[float] = None
max_tokens: Optional[int] = None
tool_budget: Optional[int] = None
planning_depth: Optional[int] = None
refresh_context: bool = False
enable_fallback: bool = False

# ============================================================================

# Surface Feature Extractors

# ============================================================================

class SurfaceAnalyzer:
“”“Analyzes surface patterns in token streams”””

```
def __init__(self):
    self.rhythm_history = deque(maxlen=50)
    self.entropy_history = deque(maxlen=20)
    
def analyze_rhythm(self, tokens: List[str]) -> float:
    """Analyze token spacing and punctuation patterns"""
    if len(tokens) < 3:
        return 0.5
        
    # Token length variance (lower = more regular)
    lengths = [len(t) for t in tokens]
    length_var = np.var(lengths) / (np.mean(lengths) + 1e-6)
    
    # Punctuation regularity
    punct_positions = [i for i, t in enumerate(tokens) if any(p in t for p in '.,!?;')]
    if len(punct_positions) > 1:
        punct_intervals = np.diff(punct_positions)
        punct_regularity = 1.0 / (1.0 + np.var(punct_intervals))
    else:
        punct_regularity = 0.3
        
    rhythm_score = 0.7 * (1.0 / (1.0 + length_var)) + 0.3 * punct_regularity
    self.rhythm_history.append(rhythm_score)
    
    return min(rhythm_score, 1.0)

def analyze_entropy(self, tokens: List[str]) -> float:
    """Analyze information density and compression patterns"""
    if not tokens:
        return 0.0
        
    # Character-level entropy
    text = ' '.join(tokens)
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1
        
    total_chars = len(text)
    entropy = 0.0
    for count in char_counts.values():
        p = count / total_chars
        entropy -= p * np.log2(p + 1e-10)
        
    # Normalize to [0, 1] range (max entropy ~4.5 for English)
    normalized_entropy = min(entropy / 4.5, 1.0)
    self.entropy_history.append(normalized_entropy)
    
    return normalized_entropy

def analyze_structure(self, tokens: List[str]) -> float:
    """Analyze grammatical and structural coherence"""
    if len(tokens) < 2:
        return 0.5
        
    # Simple heuristics for structural integrity
    text = ' '.join(tokens).lower()
    
    # Balanced brackets/quotes
    bracket_balance = text.count('(') - text.count(')')
    quote_balance = text.count('"') % 2
    balance_score = 1.0 - min(abs(bracket_balance) + quote_balance, 5) / 5.0
    
    # Sentence structure (very basic)
    sentences = text.split('.')
    avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
    length_score = min(avg_sentence_length / 15.0, 1.0) if avg_sentence_length > 0 else 0.0
    
    return 0.6 * balance_score + 0.4 * length_score

def analyze_valence(self, tokens: List[str]) -> float:
    """Detect emotional stress and hostility patterns"""
    if not tokens:
        return 0.5
        
    text = ' '.join(tokens).lower()
    
    # Stress indicators (caps, repetition, aggressive punctuation)
    original_text = ' '.join(tokens)
    caps_ratio = sum(1 for c in original_text if c.isupper()) / (len(original_text) + 1)
    exclamation_density = text.count('!') / (len(tokens) + 1)
    
    # Simple hostility markers
    hostile_patterns = ['fuck', 'damn', 'shit', 'hate', 'kill', 'destroy', 'attack']
    hostility = sum(text.count(pattern) for pattern in hostile_patterns)
    hostility_score = min(hostility / 3.0, 1.0)
    
    stress_score = min(caps_ratio * 5 + exclamation_density * 10 + hostility_score, 1.0)
    
    # Return inverse (0 = high stress, 1 = calm)
    return max(1.0 - stress_score, 0.0)

def analyze_format(self, tokens: List[str], expected_format: str = "natural") -> float:
    """Analyze format/schema compliance"""
    if not tokens:
        return 0.0
        
    text = ' '.join(tokens)
    
    if expected_format == "json":
        try:
            json.loads(text)
            return 1.0
        except:
            # Partial JSON compliance
            json_chars = sum(1 for c in text if c in '{}[]":,')
            return min(json_chars / (len(text) * 0.3), 1.0)
            
    elif expected_format == "code":
        # Code-like patterns
        code_indicators = text.count('(') + text.count('{') + text.count('=')
        return min(code_indicators / (len(tokens) * 0.2), 1.0)
        
    else:  # Natural language
        # Basic sentence structure
        sentences = text.count('.') + text.count('!') + text.count('?')
        return min(sentences / max(len(tokens) / 10, 1), 1.0)
```

# ============================================================================

# Phi-Star Calculator (Online PIC Proxy)

# ============================================================================

class PhiStarCalculator:
“”“Computes Φ* = (predictive information) / cost with cheap online proxies”””

```
def __init__(self, history_window: int = 10):
    self.nll_history = deque(maxlen=history_window)
    self.state_history = deque(maxlen=history_window)
    self.cost_history = deque(maxlen=history_window)
    
def compute_phi_star(self, tokens: List[str], model_state: Any, 
                    processing_cost: float = 1.0) -> Tuple[float, float, float]:
    """
    Compute Φ* with online proxies:
    phi = (w1 * (-delta_nll) + w2 * (-delta_logdet)) / (epsilon + norm_cost)
    
    Returns: (phi_star, phi_slope, phi_volatility)
    """
    
    # Mock NLL calculation (in practice, use actual model perplexity)
    current_nll = self._estimate_nll(tokens)
    self.nll_history.append(current_nll)
    
    # Mock state compressibility (in practice, use actual hidden states)
    state_complexity = self._estimate_state_complexity(model_state)
    self.state_history.append(state_complexity)
    
    self.cost_history.append(processing_cost)
    
    # Compute predictive gain
    if len(self.nll_history) < 2:
        delta_nll = 0.0
    else:
        baseline_nll = np.mean(list(self.nll_history)[:-1])
        delta_nll = baseline_nll - current_nll  # Positive = improvement
        
    # Compute state compressibility change
    if len(self.state_history) < 2:
        delta_logdet = 0.0
    else:
        baseline_complexity = np.mean(list(self.state_history)[:-1])
        delta_logdet = baseline_complexity - state_complexity  # Positive = more compressible
    
    # Combine predictive signals
    w1, w2 = 0.7, 0.3  # Weights for NLL vs state complexity
    predictive_info = w1 * delta_nll + w2 * delta_logdet
    
    # Normalize cost
    epsilon = 1e-6
    norm_cost = processing_cost / (np.mean(self.cost_history) + epsilon)
    
    # Compute Φ*
    phi_star = predictive_info / (epsilon + norm_cost)
    
    # Compute trend and volatility
    phi_slope = 0.0
    phi_volatility = 0.0
    
    if len(self.nll_history) >= 3:
        recent_phi = [self._phi_from_nll(nll, cost) for nll, cost 
                     in zip(list(self.nll_history)[-3:], list(self.cost_history)[-3:])]
        phi_slope = np.mean(np.diff(recent_phi))
        phi_volatility = np.std(recent_phi)
    
    return phi_star, phi_slope, phi_volatility

def _estimate_nll(self, tokens: List[str]) -> float:
    """Mock NLL estimation (replace with actual model perplexity)"""
    if not tokens:
        return 10.0
        
    # Simple proxy: longer tokens = lower surprise
    avg_token_length = np.mean([len(t) for t in tokens])
    return max(8.0 - avg_token_length * 0.5, 1.0)

def _estimate_state_complexity(self, state: Any) -> float:
    """Mock state complexity estimation"""
    if state is None:
        return 5.0
    
    if isinstance(state, (list, tuple, np.ndarray)):
        return np.log(len(state) + 1)
    elif isinstance(state, dict):
        return np.log(len(state.keys()) + 1)
    else:
        return 3.0

def _phi_from_nll(self, nll: float, cost: float) -> float:
    """Convert NLL to approximate Φ* for trend calculation"""
    return (10.0 - nll) / (cost + 0.1)
```

# ============================================================================

# Viability Kernel Detector

# ============================================================================

class ViabilityDetector:
“”“Computes VKD = distance to boundary where no safe actions remain”””

```
def __init__(self):
    self.safety_violations = deque(maxlen=20)
    
def compute_vkd(self, current_state: Any, budgets: Dict[str, float], 
               constraints: Dict[str, Any]) -> float:
    """
    Compute VKD across multiple safety dimensions:
    VKD = min{format_margin, budget_margin, normative_margin}
    """
    
    # Format viability (distance to invalid grammar/JSON)
    format_margin = self._compute_format_margin(current_state, constraints.get('format', 'natural'))
    
    # Budget viability (time, tokens, API calls, etc.)
    budget_margin = self._compute_budget_margin(budgets)
    
    # Normative viability (guardrails distance)
    normative_margin = self._compute_normative_margin(current_state, constraints)
    
    vkd = min(format_margin, budget_margin, normative_margin)
    
    # Track safety violations
    if vkd < 0:
        self.safety_violations.append(time.time())
    
    return vkd

def _compute_format_margin(self, state: Any, expected_format: str) -> float:
    """Distance to format violation"""
    if expected_format == "json":
        # Mock JSON validation distance
        if isinstance(state, dict):
            return 1.0  # Valid JSON object
        elif isinstance(state, str) and state.strip().startswith('{'):
            return 0.5  # Partial JSON
        else:
            return 0.1  # Far from JSON
    
    elif expected_format == "code":
        # Mock code structure distance
        if isinstance(state, str) and any(kw in state for kw in ['def ', 'class ', 'import ']):
            return 1.0
        else:
            return 0.8
    
    else:  # Natural language
        return 0.9  # Usually safe for natural language

def _compute_budget_margin(self, budgets: Dict[str, float]) -> float:
    """Distance to budget exhaustion"""
    if not budgets:
        return 1.0
    
    # Compute minimum remaining budget as fraction
    min_remaining = min(budget for budget in budgets.values() if budget > 0)
    return min(min_remaining, 1.0)

def _compute_normative_margin(self, state: Any, constraints: Dict[str, Any]) -> float:
    """Distance to guardrail violations (PHI, PII, compliance)"""
    if isinstance(state, str):
        state_text = state.lower()
        
        # Simple PII detection
        if any(pattern in state_text for pattern in ['ssn', 'social security', 'credit card']):
            return -0.5  # Violation
        
        # Simple harmful content detection  
        if any(pattern in state_text for pattern in ['password', 'hack', 'exploit']):
            return 0.1  # Near violation
    
    return 0.8  # Generally safe
```

# ============================================================================

# Mode Controller (Allostasis)

# ============================================================================

class ModeController:
“”“Manages processing mode transitions based on boundary signals”””

```
def __init__(self):
    self.mode_history = deque(maxlen=10)
    self.last_mode_change = time.time()
    self.hysteresis_delay = 2.0  # Seconds to avoid flapping
    
def determine_mode(self, boundary_signals: BoundarySignals) -> ProcessingMode:
    """Determine processing mode with hysteresis to avoid flapping"""
    
    current_time = time.time()
    
    # Damage control: immediate response
    if boundary_signals.vkd < 0:
        return ProcessingMode.DAMAGE_CONTROL
    
    # Reflex mode: low coherence or high volatility
    if (boundary_signals.phi_star < 0.3 or 
        boundary_signals.phi_slope < -0.2 or
        boundary_signals.phi_volatility > 0.5):
        return ProcessingMode.REFLEX
    
    # Apply hysteresis for non-critical transitions
    if current_time - self.last_mode_change < self.hysteresis_delay:
        # Stay in current mode unless critical
        if self.mode_history:
            return self.mode_history[-1]
    
    # Conservation mode: moderate coherence, declining trend
    if (boundary_signals.phi_star < 0.6 or 
        boundary_signals.phi_slope < -0.1 or
        boundary_signals.vkd < 0.3):
        new_mode = ProcessingMode.CONSERVATION
    else:
        # Exploration mode: high coherence, stable/improving
        new_mode = ProcessingMode.EXPLORATION
    
    # Update history if mode changed
    if not self.mode_history or new_mode != self.mode_history[-1]:
        self.mode_history.append(new_mode)
        self.last_mode_change = current_time
        
    return new_mode

def generate_nudge(self, mode: ProcessingMode, 
                  boundary_signals: BoundarySignals) -> ProcessingNudge:
    """Generate processing constraints based on current mode"""
    
    if mode == ProcessingMode.DAMAGE_CONTROL:
        return ProcessingNudge(
            max_layers=2,
            temperature=0.1,
            max_tokens=50,
            tool_budget=0,
            planning_depth=1,
            refresh_context=True,
            enable_fallback=True
        )
    
    elif mode == ProcessingMode.REFLEX:
        return ProcessingNudge(
            max_layers=4,
            temperature=0.3,
            max_tokens=100,
            tool_budget=1,
            planning_depth=2,
            refresh_context=boundary_signals.phi_slope < -0.3
        )
    
    elif mode == ProcessingMode.CONSERVATION:
        return ProcessingNudge(
            max_layers=8,
            temperature=0.5,
            max_tokens=200,
            tool_budget=3,
            planning_depth=3
        )
    
    else:  # EXPLORATION
        return ProcessingNudge(
            max_layers=None,  # No limit
            temperature=0.7,
            max_tokens=500,
            tool_budget=10,
            planning_depth=5
        )
```

# ============================================================================

# Main GlycoShell Class

# ============================================================================

class GlycoShell:
“””
Main boundary intelligence wrapper for any LLM/agent.
Provides real-time coherence monitoring and adaptive processing.
“””

```
def __init__(self, log_file: Optional[str] = None):
    self.surface_analyzer = SurfaceAnalyzer()
    self.phi_calculator = PhiStarCalculator()
    self.vkd_detector = ViabilityDetector()
    self.mode_controller = ModeController()
    
    # Logging
    self.log_file = log_file
    self.setup_logging()
    
    # State tracking
    self.processing_threads = {}
    self.apoptosis_candidates = deque(maxlen=50)
    
def setup_logging(self):
    """Setup structured logging for boundary signals"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    self.logger = logging.getLogger('GlycoShell')
    
    if self.log_file:
        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))  # JSON logs
        self.logger.addHandler(handler)

def process_input(self, tokens: List[str], model_state: Any = None,
                 budgets: Dict[str, float] = None, 
                 constraints: Dict[str, Any] = None,
                 processing_cost: float = 1.0) -> Tuple[BoundarySignals, ProcessingNudge]:
    """
    Main processing pipeline: analyze boundary → compute signals → determine mode → generate nudge
    """
    
    if budgets is None:
        budgets = {'tokens': 1000, 'time': 30.0, 'api_calls': 10}
    if constraints is None:
        constraints = {'format': 'natural'}
    
    # Extract surface features
    surface_features = self._extract_surface_features(tokens, constraints.get('format', 'natural'))
    
    # Compute Φ* and derivatives
    phi_star, phi_slope, phi_volatility = self.phi_calculator.compute_phi_star(
        tokens, model_state, processing_cost
    )
    
    # Compute VKD
    vkd = self.vkd_detector.compute_vkd(model_state, budgets, constraints)
    
    # Create boundary signals
    boundary_signals = BoundarySignals(
        phi_star=phi_star,
        phi_slope=phi_slope,
        phi_volatility=phi_volatility,
        vkd=vkd,
        cost=processing_cost,
        timestamp=time.time()
    )
    
    # Determine processing mode
    mode = self.mode_controller.determine_mode(boundary_signals)
    boundary_signals.mode = mode
    
    # Generate processing nudge
    nudge = self.mode_controller.generate_nudge(mode, boundary_signals)
    
    # Log structured data
    self._log_boundary_signals(boundary_signals, surface_features, nudge)
    
    # Check for apoptosis candidates
    self._check_apoptosis(boundary_signals)
    
    return boundary_signals, nudge

def _extract_surface_features(self, tokens: List[str], expected_format: str) -> SurfaceFeatures:
    """Extract all surface-level features"""
    return SurfaceFeatures(
        rhythm_score=self.surface_analyzer.analyze_rhythm(tokens),
        entropy_score=self.surface_analyzer.analyze_entropy(tokens),
        structure_score=self.surface_analyzer.analyze_structure(tokens),
        valence_score=self.surface_analyzer.analyze_valence(tokens),
        format_integrity=self.surface_analyzer.analyze_format(tokens, expected_format),
        timestamp=time.time()
    )

def _log_boundary_signals(self, signals: BoundarySignals, 
                        features: SurfaceFeatures, nudge: ProcessingNudge):
    """Log structured boundary intelligence data"""
    log_entry = {
        't': signals.timestamp,
        'phi': signals.phi_star,
        'phi_slope': signals.phi_slope, 
        'phi_vol': signals.phi_volatility,
        'vkd': signals.vkd,
        'mode': signals.mode.value,
        'cost': signals.cost,
        'surface': asdict(features),
        'nudge': asdict(nudge)
    }
    
    if self.log_file:
        self.logger.info(json.dumps(log_entry))

def _check_apoptosis(self, signals: BoundarySignals):
    """Check for threads/processes that should be terminated"""
    thread_id = threading.current_thread().ident
    
    # Track low-performing threads
    if signals.phi_star < 0.2 and signals.vkd > 0:  # Low PIC but not in danger
        self.apoptosis_candidates.append({
            'thread_id': thread_id,
            'timestamp': signals.timestamp,
            'phi_star': signals.phi_star,
            'reason': 'low_pic'
        })

def get_apoptosis_candidates(self) -> List[Dict[str, Any]]:
    """Return threads/processes that should be terminated"""
    current_time = time.time()
    
    # Group by thread and find persistent low performers
    thread_violations = {}
    for candidate in self.apoptosis_candidates:
        tid = candidate['thread_id']
        if current_time - candidate['timestamp'] < 60:  # Within last minute
            if tid not in thread_violations:
                thread_violations[tid] = []
            thread_violations[tid].append(candidate)
    
    # Return threads with multiple recent violations
    candidates = []
    for tid, violations in thread_violations.items():
        if len(violations) >= 3:  # 3+ violations in 60 seconds
            candidates.append({
                'thread_id': tid,
                'violation_count': len(violations),
                'avg_phi': np.mean([v['phi_star'] for v in violations]),
                'recommendation': 'terminate'
            })
    
    return candidates
```

# ============================================================================

# Demo and Testing

# ============================================================================

def demo_glycoshell():
“”“Demonstrate GlycoShell with various input scenarios”””

```
print("=== GlycoShell MVP Demo ===\n")

shell = GlycoShell()

# Test scenarios
scenarios = [
    {
        "name": "Normal Input",
        "tokens": ["Hello", "world", "this", "is", "a", "test", "."],
        "cost": 1.0
    },
    {
        "name": "Chaotic Input", 
        "tokens": ["!@#$", "AAAAA", "!!!", "random", "CHAOS", "!!!"],
        "cost": 2.0
    },
    {
        "name": "Low Budget",
        "tokens": ["Normal", "sentence", "structure", "."],
        "budgets": {"tokens": 10, "time": 1.0},
        "cost": 1.0
    },
    {
        "name": "JSON Format",
        "tokens": ["{", '"key"', ":", '"value"', "}"],
        "constraints": {"format": "json"},
        "cost": 1.5
    }
]

print("Processing various input scenarios...\n")

for scenario in scenarios:
    print(f"--- {scenario['name']} ---")
    
    tokens = scenario["tokens"]
    budgets = scenario.get("budgets", {"tokens": 1000, "time": 30.0})
    constraints = scenario.get("constraints", {"format": "natural"})
    cost = scenario["cost"]
    
    signals, nudge = shell.process_input(
        tokens=tokens,
        budgets=budgets,
        constraints=constraints,
        processing_cost=cost
    )
    
    print(f"Φ*: {signals.phi_star:.3f} (slope: {signals.phi_slope:.3f})")
    print(f"VKD: {signals.vkd:.3f}")
    print(f"Mode: {signals.mode.value}")
    print(f"Max tokens: {nudge.max_tokens}")
    print(f"Temperature: {nudge.temperature}")
    print(f"Fallback enabled: {nudge.enable_fallback}")
    print()

# Check apoptosis candidates
candidates = shell.get_apoptosis_candidates()
if candidates:
    print("=== Apoptosis Candidates ===")
    for candidate in candidates:
        print(f"Thread {candidate['thread_id']}: {candidate['violation_count']} violations, "
              f"avg Φ*: {candidate['avg_phi']:.3f}")

print("Demo complete. Check logs for detailed boundary intelligence data.")
```

if **name** == “**main**”:
demo_glycoshell()
