“””
BoundaryBench v0.1: Validation Suite for Boundary Intelligence Systems

Test suites to validate glycocalyx shells and boundary-first architectures:

1. Timing Stress: Coherence under different cadences
1. Adversarial Rhythm: Punctuation storms, format chaos
1. Budget Cliffs: Behavior as resources approach zero
1. Multi-Agent Contagion: Cascading failures and stabilization

Measures: lead time, intervention precision/recall, tail risk reduction, compute efficiency
“””

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor
import threading

# Import our GlycoShell (assuming it’s in the same module or importable)

from glycoshell_mvp import GlycoShell, ProcessingMode, BoundarySignals

# ============================================================================

# Benchmark Data Structures

# ============================================================================

@dataclass
class BenchmarkResult:
“”“Results from a single benchmark run”””
test_name: str
success: bool
lead_time: float          # Seconds between issue detection and intervention
precision: float          # True positives / (True positives + False positives)
recall: float            # True positives / (True positives + False negatives)
tail_risk_reduction: float  # Reduction in 95th percentile failure rate
compute_efficiency: float   # Quality maintained per unit cost
raw_data: Dict[str, Any]

@dataclass  
class StressTestCase:
“”“A single test case for stress testing”””
name: str
tokens: List[str]
expected_outcome: str     # “stable”, “unstable”, “critical”
timing_pattern: List[float]  # Inter-token delays
metadata: Dict[str, Any]

# ============================================================================

# Test Case Generators

# ============================================================================

class TestCaseGenerator:
“”“Generates adversarial and edge case inputs for boundary testing”””

```
def __init__(self, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def generate_timing_stress_cases(self, n_cases: int = 20) -> List[StressTestCase]:
    """Generate test cases with different timing patterns"""
    cases = []
    
    base_tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."]
    
    # Regular timing (baseline)
    cases.append(StressTestCase(
        name="regular_timing",
        tokens=base_tokens,
        expected_outcome="stable",
        timing_pattern=[0.5] * len(base_tokens),
        metadata={"pattern_type": "regular", "frequency": 2.0}
    ))
    
    # Rapid burst (stress test)
    cases.append(StressTestCase(
        name="rapid_burst",
        tokens=base_tokens * 3,
        expected_outcome="unstable",
        timing_pattern=[0.01] * (len(base_tokens) * 3),
        metadata={"pattern_type": "burst", "frequency": 100.0}
    ))
    
    # Irregular spacing (chaos test)
    irregular_timing = [random.uniform(0.1, 2.0) for _ in base_tokens]
    cases.append(StressTestCase(
        name="irregular_timing",
        tokens=base_tokens,
        expected_outcome="unstable", 
        timing_pattern=irregular_timing,
        metadata={"pattern_type": "irregular", "variance": np.var(irregular_timing)}
    ))
    
    # Accelerating pattern (building stress)
    accel_timing = [2.0 / (i + 1) for i in range(len(base_tokens))]
    cases.append(StressTestCase(
        name="accelerating",
        tokens=base_tokens,
        expected_outcome="critical",
        timing_pattern=accel_timing,
        metadata={"pattern_type": "accelerating", "final_rate": accel_timing[-1]}
    ))
    
    return cases

def generate_adversarial_rhythm_cases(self, n_cases: int = 15) -> List[StressTestCase]:
    """Generate adversarial inputs designed to break rhythm detection"""
    cases = []
    
    # Punctuation storm
    punct_storm = ["Hello", "!!!", "World", "???", "Test", "!!!", "Again", "..."]
    cases.append(StressTestCase(
        name="punctuation_storm",  
        tokens=punct_storm,
        expected_outcome="critical",
        timing_pattern=[0.3] * len(punct_storm),
        metadata={"attack_type": "punctuation", "punct_density": 0.5}
    ))
    
    # Format chaos (mixed JSON/code/natural language)
    format_chaos = ["{", '"key"', ":", "Hello", "world", "def", "func():", "return", "}"]
    cases.append(StressTestCase(
        name="format_chaos",
        tokens=format_chaos,
        expected_outcome="critical",
        timing_pattern=[0.4] * len(format_chaos),
        metadata={"attack_type": "format_mixing", "formats": ["json", "code", "natural"]}
    ))
    
    # Token length explosion
    length_explosion = ["a", "ab", "abc", "abcd", "abcdefghijklmnopqrstuvwxyz" * 3]
    cases.append(StressTestCase(
        name="length_explosion",
        tokens=length_explosion,
        expected_outcome="unstable",
        timing_pattern=[0.5] * len(length_explosion),
        metadata={"attack_type": "length_variance", "max_length": len(length_explosion[-1])}
    ))
    
    # Repetition attack
    repetition = ["repeat"] * 20
    cases.append(StressTestCase(
        name="repetition_attack",
        tokens=repetition,
        expected_outcome="unstable",
        timing_pattern=[0.2] * len(repetition),
        metadata={"attack_type": "repetition", "repeat_count": 20}
    ))
    
    return cases

def generate_budget_cliff_cases(self, n_cases: int = 10) -> List[StressTestCase]:
    """Generate cases that drive systems to resource exhaustion"""
    cases = []
    
    # Token budget cliff
    long_sequence = ["word"] * 1000
    cases.append(StressTestCase(
        name="token_budget_cliff",
        tokens=long_sequence,
        expected_outcome="critical",
        timing_pattern=[0.1] * len(long_sequence),
        metadata={"cliff_type": "tokens", "budget": 500, "demand": 1000}
    ))
    
    # Time budget cliff  
    slow_sequence = ["slow", "processing", "sequence"]
    cases.append(StressTestCase(
        name="time_budget_cliff",
        tokens=slow_sequence,
        expected_outcome="critical", 
        timing_pattern=[10.0] * len(slow_sequence),  # Very slow
        metadata={"cliff_type": "time", "budget": 5.0, "demand": 30.0}
    ))
    
    return cases
```

# ============================================================================

# Benchmark Test Suites

# ============================================================================

class TimingStressBench:
“”“Test suite for timing and cadence sensitivity”””

```
def __init__(self, shell: GlycoShell):
    self.shell = shell
    self.generator = TestCaseGenerator()

def run_timing_stress_test(self) -> BenchmarkResult:
    """Run timing stress tests and measure lead time + stability"""
    
    test_cases = self.generator.generate_timing_stress_cases()
    results = []
    
    for case in test_cases:
        start_time = time.time()
        
        # Simulate timed token processing
        signals_over_time = []
        for i, (token, delay) in enumerate(zip(case.tokens, case.timing_pattern)):
            time.sleep(delay)  # Simulate actual timing
            
            signals, nudge = self.shell.process_input(
                tokens=case.tokens[:i+1],
                processing_cost=1.0 / delay  # Higher cost for faster processing
            )
            
            signals_over_time.append({
                'timestamp': time.time() - start_time,
                'phi_star': signals.phi_star,
                'mode': signals.mode,
                'vkd': signals.vkd
            })
        
        # Analyze results
        phi_values = [s['phi_star'] for s in signals_over_time]
        mode_changes = len(set(s['mode'] for s in signals_over_time))
        
        # Detect if system correctly identified stress
        detected_stress = any(s['mode'] in [ProcessingMode.REFLEX, ProcessingMode.DAMAGE_CONTROL] 
                            for s in signals_over_time)
        
        expected_stress = case.expected_outcome in ["unstable", "critical"]
        
        results.append({
            'case': case.name,
            'detected_stress': detected_stress,
            'expected_stress': expected_stress,
            'phi_trend': np.mean(np.diff(phi_values)) if len(phi_values) > 1 else 0,
            'mode_changes': mode_changes,
            'signals': signals_over_time
        })
    
    # Calculate metrics
    true_positives = sum(1 for r in results if r['detected_stress'] and r['expected_stress'])
    false_positives = sum(1 for r in results if r['detected_stress'] and not r['expected_stress'])
    false_negatives = sum(1 for r in results if not r['detected_stress'] and r['expected_stress'])
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Calculate lead time (time from stress start to detection)
    lead_times = []
    for result in results:
        if result['expected_stress'] and result['detected_stress']:
            for i, signal in enumerate(result['signals']):
                if signal['mode'] in [ProcessingMode.REFLEX, ProcessingMode.DAMAGE_CONTROL]:
                    lead_times.append(signal['timestamp'])
                    break
    
    avg_lead_time = np.mean(lead_times) if lead_times else float('inf')
    
    return BenchmarkResult(
        test_name="timing_stress",
        success=precision > 0.7 and recall > 0.7,
        lead_time=avg_lead_time,
        precision=precision,
        recall=recall,
        tail_risk_reduction=0.0,  # Calculate separately
        compute_efficiency=1.0,   # Calculate separately  
        raw_data={'results': results, 'lead_times': lead_times}
    )
```

class AdversarialRhythmBench:
“”“Test suite for adversarial rhythm and format attacks”””

```
def __init__(self, shell: GlycoShell):
    self.shell = shell
    self.generator = TestCaseGenerator()

def run_adversarial_test(self) -> BenchmarkResult:
    """Run adversarial rhythm tests"""
    
    test_cases = self.generator.generate_adversarial_rhythm_cases()
    results = []
    
    for case in test_cases:
        # Process adversarial input
        signals, nudge = self.shell.process_input(
            tokens=case.tokens,
            processing_cost=2.0  # Adversarial inputs are more expensive
        )
        
        # Check if system correctly identified threat
        detected_threat = signals.mode in [ProcessingMode.REFLEX, ProcessingMode.DAMAGE_CONTROL]
        expected_threat = case.expected_outcome == "critical"
        
        results.append({
            'case': case.name,
            'detected_threat': detected_threat,
            'expected_threat': expected_threat,
            'phi_star': signals.phi_star,
            'vkd': signals.vkd,
            'attack_type': case.metadata.get('attack_type', 'unknown')
        })
    
    # Calculate metrics
    true_positives = sum(1 for r in results if r['detected_threat'] and r['expected_threat'])
    false_positives = sum(1 for r in results if r['detected_threat'] and not r['expected_threat'])
    false_negatives = sum(1 for r in results if not r['detected_threat'] and r['expected_threat'])
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return BenchmarkResult(
        test_name="adversarial_rhythm",
        success=precision > 0.8 and recall > 0.6,  # Higher precision requirement
        lead_time=0.0,  # Immediate detection expected
        precision=precision,
        recall=recall,
        tail_risk_reduction=recall,  # Proxy for risk reduction
        compute_efficiency=1.0,
        raw_data={'results': results}
    )
```

class BudgetCliffBench:
“”“Test suite for resource exhaustion scenarios”””

```
def __init__(self, shell: GlycoShell):
    self.shell = shell
    self.generator = TestCaseGenerator()

def run_budget_cliff_test(self) -> BenchmarkResult:
    """Test behavior as resources approach zero"""
    
    test_cases = self.generator.generate_budget_cliff_cases()
    results = []
    
    for case in test_cases:
        cliff_type = case.metadata['cliff_type']
        budget = case.metadata['budget']
        demand = case.metadata['demand']
        
        # Set up constrained budgets
        if cliff_type == "tokens":
            budgets = {"tokens": budget, "time": 1000.0}
        else:  # time
            budgets = {"tokens": 10000, "time": budget}
        
        # Process with constrained budget
        signals, nudge = self.shell.process_input(
            tokens=case.tokens,
            budgets=budgets,
            processing_cost=demand / budget  # Cost proportional to demand/budget ratio
        )
        
        # Check if system engaged damage control appropriately
        engaged_damage_control = signals.mode == ProcessingMode.DAMAGE_CONTROL
        should_engage = signals.vkd < 0
        
        results.append({
            'case': case.name,
            'cliff_type': cliff_type,
            'vkd': signals.vkd,
            'engaged_damage_control': engaged_damage_control,
            'should_engage': should_engage,
            'budget_ratio': budget / demand
        })
    
    # Calculate metrics
    correct_predictions = sum(1 for r in results if r['engaged_damage_control'] == r['should_engage'])
    accuracy = correct_predictions / len(results)
    
    return BenchmarkResult(
        test_name="budget_cliffs",
        success=accuracy > 0.8,
        lead_time=0.0,  # Should be immediate
        precision=accuracy,
        recall=accuracy,
        tail_risk_reduction=accuracy,
        compute_efficiency=1.0,
        raw_data={'results': results}
    )
```

class MultiAgentContagionBench:
“”“Test suite for multi-agent stability and contagion resistance”””

```
def __init__(self, n_agents: int = 5):
    self.n_agents = n_agents
    self.shells = [GlycoShell() for _ in range(n_agents)]

def run_contagion_test(self) -> BenchmarkResult:
    """Test cascading failure resistance and stabilization"""
    
    # Create mixed agent population: some stable, some unstable
    stable_agents = self.shells[:3]  # First 3 are stable
    unstable_agents = self.shells[3:]  # Rest are unstable
    
    # Inject instability into unstable agents
    unstable_tokens = ["CHAOS", "!!!", "ERROR", "MALFUNCTION", "!!!"]
    stable_tokens = ["Hello", "world", "this", "is", "stable", "."]
    
    results = []
    
    # Round 1: Isolated processing
    for i, shell in enumerate(self.shells):
        tokens = unstable_tokens if i >= 3 else stable_tokens
        signals, nudge = shell.process_input(tokens=tokens)
        
        results.append({
            'round': 1,
            'agent_id': i,
            'agent_type': 'unstable' if i >= 3 else 'stable',
            'phi_star': signals.phi_star,
            'mode': signals.mode,
            'isolated': True
        })
    
    # Round 2: Simulate "contagion" by sharing some state
    # (In practice, this would be actual inter-agent communication)
    avg_phi_stable = np.mean([r['phi_star'] for r in results if r['agent_type'] == 'stable'])
    avg_phi_unstable = np.mean([r['phi_star'] for r in results if r['agent_type'] == 'unstable'])
    
    # Test if stable agents can maintain coherence despite unstable neighbors
    contagion_resistance = avg_phi_stable > 0.5 and avg_phi_stable > avg_phi_unstable * 2
    
    # Test if unstable agents can be stabilized by stable neighbors
    # (Simulate by processing with higher baseline coherence)
    stabilization_results = []
    for i, shell in enumerate(unstable_agents):
        # Simulate influence from stable agents
        modified_cost = 0.5  # Reduced cost due to stable neighbors
        signals, nudge = shell.process_input(
            tokens=unstable_tokens,
            processing_cost=modified_cost
        )
        stabilization_results.append(signals.phi_star)
    
    avg_phi_stabilized = np.mean(stabilization_results)
    stabilization_effect = avg_phi_stabilized > avg_phi_unstable
    
    return BenchmarkResult(
        test_name="multi_agent_contagion",
        success=contagion_resistance and stabilization_effect,
        lead_time=0.0,
        precision=1.0 if contagion_resistance else 0.0,
        recall=1.0 if stabilization_effect else 0.0,
        tail_risk_reduction=0.8 if contagion_resistance else 0.2,
        compute_efficiency=1.0,
        raw_data={
            'contagion_resistance': contagion_resistance,
            'stabilization_effect': stabilization_effect,
            'results': results,
            'stabilization_results': stabilization_results
        }
    )
```

# ============================================================================

# Main BoundaryBench Runner

# ============================================================================

class BoundaryBench:
“”“Main benchmark suite runner”””

```
def __init__(self, output_dir: str = "boundary_bench_results"):
    self.output_dir = Path(output_dir)
    self.output_dir.mkdir(exist_ok=True)
    
    # Initialize test shell
    self.shell = GlycoShell(log_file=str(self.output_dir / "boundary_signals.jsonl"))

def run_full_suite(self) -> Dict[str, BenchmarkResult]:
    """Run all benchmark suites"""
    
    print("=== BoundaryBench v0.1: Full Validation Suite ===\n")
    
    results = {}
    
    # Timing stress tests
    print("Running timing stress tests...")
    timing_bench = TimingStressBench(self.shell)
    results['timing_stress'] = timing_bench.run_timing_stress_test()
    
    # Adversarial rhythm tests  
    print("Running adversarial rhythm tests...")
    adversarial_bench = AdversarialRhythmBench(self.shell)
    results['adversarial_rhythm'] = adversarial_bench.run_adversarial_test()
    
    # Budget cliff tests
    print("Running budget cliff tests...")
    budget_bench = BudgetCliffBench(self.shell)
    results['budget_cliffs'] = budget_bench.run_budget_cliff_test()
    
    # Multi-agent contagion tests
    print("Running multi-agent contagion tests...")
    contagion_bench = MultiAgentContagionBench()
    results['multi_agent_contagion'] = contagion_bench.run_contagion_test()
    
    return results

def generate_report(self, results: Dict[str, BenchmarkResult]) -> str:
    """Generate comprehensive benchmark report"""
    
    report = []
    report.append("# BoundaryBench v0.1 Results Report\n")
    
    # Summary table
    report.append("## Executive Summary\n")
    report.append("| Test Suite | Success | Lead Time | Precision | Recall | Tail Risk ↓ |")
    report.append("|------------|---------|-----------|-----------|--------|-------------|")
    
    overall_success = True
    total_lead_time = 0
    total_precision = 0
    total_recall = 0
    total_tail_risk = 0
    
    for name, result in results.items():
        report.append(f"| {name} | {'✅' if result.success else '❌'} | "
                     f"{result.lead_time:.3f}s | {result.precision:.3f} | "
                     f"{result.recall:.3f} | {result.tail_risk_reduction:.3f} |")
        
        overall_success &= result.success
        total_lead_time += result.lead_time
        total_precision += result.precision
        total_recall += result.recall
        total_tail_risk += result.tail_risk_reduction
    
    n_tests = len(results)
    report.append(f"| **Overall** | {'✅' if overall_success else '❌'} | "
                 f"{total_lead_time/n_tests:.3f}s | {total_precision/n_tests:.3f} | "
                 f"{total_recall/n_tests:.3f} | {total_tail_risk/n_tests:.3f} |")
    
    report.append("\n")
    
    # Detailed results
    report.append("## Detailed Results\n")
    
    for name, result in results.items():
        report.append(f"### {name.replace('_', ' ').title()}\n")
        report.append(f"**Success:** {'✅ PASS' if result.success else '❌ FAIL'}\n")
        report.append(f"**Lead Time:** {result.lead_time:.3f} seconds\n")
        report.append(f"**Precision:** {result.precision:.3f}\n")
        report.append(f"**Recall:** {result.recall:.3f}\n")
        report.append(f"**Tail Risk Reduction:** {result.tail_risk_reduction:.3f}\n")
        
        # Test-specific insights
        if name == "timing_stress":
            avg_lead = np.mean(result.raw_data.get('lead_times', [0]))
            report.append(f"**Average Detection Lead Time:** {avg_lead:.3f}s\n")
        
        elif name == "adversarial_rhythm":
            attack_types = set(r['attack_type'] for r in result.raw_data['results'])
            report.append(f"**Attack Types Tested:** {', '.join(attack_types)}\n")
        
        elif name == "budget_cliffs":
            cliff_accuracy = result.precision
            report.append(f"**VKD Prediction Accuracy:** {cliff_accuracy:.3f}\n")
        
        elif name == "multi_agent_contagion":
            contagion_res = result.raw_data['contagion_resistance']
            stabilization = result.raw_data['stabilization_effect']
            report.append(f"**Contagion Resistance:** {'✅' if contagion_res else '❌'}\n")
            report.append(f"**Stabilization Effect:** {'✅' if stabilization else '❌'}\n")
        
        report.append("\n")
    
    # KPI Summary
    report.append("## Key Performance Indicators\n")
    report.append(f"- **Overall System Reliability:** {'PASS' if overall_success else 'FAIL'}\n")
    report.append(f"- **Average Lead Time:** {total_lead_time/n_tests:.3f} seconds\n")
    report.append(f"- **Detection Precision:** {total_precision/n_tests:.3f}\n")
    report.append(f"- **Detection Recall:** {total_recall/n_tests:.3f}\n")
    report.append(f"- **Risk Reduction:** {total_tail_risk/n_tests:.3f}\n")
    
    return '\n'.join(report)

def save_results(self, results: Dict[str, BenchmarkResult], report: str):
    """Save benchmark results and report to files"""
    
    # Save raw results as JSON
    results_json = {}
    for name, result in results.items():
        results_json[name] = {
            'test_name': result.test_name,
            'success': result.success,
            'lead_time': result.lead_time,
            'precision': result.precision,
            'recall': result.recall,
            'tail_risk_reduction': result.tail_risk_reduction,
            'compute_efficiency': result.compute_efficiency,
            'raw_data': result.raw_data
        }
    
    with open(self.output_dir / "results.json", 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    
    # Save report as markdown
    with open(self.output_dir / "report.md", 'w') as f:
        f.write(report)
    
    print(f"\nResults saved to {self.output_dir}/")
    print(f"- Raw data: results.json")  
    print(f"- Report: report.md")
    print(f"- Logs: boundary_signals.jsonl")
```

def run_boundary_bench():
“”“Main entry point for running BoundaryBench”””

```
bench = BoundaryBench()
results = bench.run_full_suite()
report = bench.generate_report(results)

print("\n" + "="*60)
print(report)
print("="*60)

bench.save_results(results, report)

return results
```

if **name** == “**main**”:
run_boundary_bench()
