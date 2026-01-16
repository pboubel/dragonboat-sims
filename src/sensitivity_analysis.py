"""
Sensitivity Analysis for Dragon Boat Second-Order Effects

This script explores how the simulation results vary across the plausible
parameter space. Given the weak empirical foundation for several parameters,
this analysis is essential for understanding the reliability of our conclusions.

Key questions:
1. How sensitive is the sync vs. desync comparison to each parameter?
2. What parameter ranges would flip the conclusion?
3. Which parameters most urgently need experimental validation?
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json

# Import the simulation (assumes dragon_boat_enhanced.py is available)
from dragon_boat_enhanced import simulate_enhanced, EnhancedBoatParams, EnhancedPaddleParams


# =============================================================================
# Parameter Ranges for Sensitivity Analysis
# =============================================================================

# Each parameter has: (name, baseline, low, high, confidence, description)
PARAMETER_RANGES = {
    # Wake parameters
    'wake_decay_time': {
        'baseline': 0.3,
        'low': 0.1,
        'high': 0.8,
        'confidence': 'medium',
        'description': 'Time for wake to decay (seconds)',
        'source': 'Sliasas & Tullis (2009) suggest 0.2-0.5s for rowing blades'
    },
    'wake_decay_distance': {
        'baseline': 0.5,
        'low': 0.2,
        'high': 1.5,
        'confidence': 'low',
        'description': 'Spatial decay length scale (meters)',
        'source': 'No direct measurement; estimated from blade width'
    },
    'wake_max_penalty': {
        'baseline': 0.15,
        'low': 0.05,
        'high': 0.30,
        'confidence': 'low',
        'description': 'Maximum efficiency reduction from wake',
        'source': 'Arbitrary assumption - no empirical basis'
    },
    
    # Coupling parameters (MOST UNCERTAIN)
    'coupling_strength': {
        'baseline': 0.05,
        'low': 0.01,
        'high': 0.15,
        'confidence': 'very_low',
        'description': 'Max efficiency reduction per nearby blade',
        'source': 'Extrapolated from propeller studies with different geometry'
    },
    'coupling_distance': {
        'baseline': 1.2,
        'low': 0.5,
        'high': 3.0,
        'confidence': 'very_low',
        'description': 'Interaction length scale (meters)',
        'source': 'Arbitrary assumption'
    },
    'coupling_opposite_ratio': {
        'baseline': 0.3,
        'low': 0.0,
        'high': 0.8,
        'confidence': 'very_low',
        'description': 'Coupling strength for opposite-side paddles relative to same-side',
        'source': 'Geometric guess'
    },
    
    # Added mass parameters
    'added_mass_coeff': {
        'baseline': 0.10,
        'low': 0.05,
        'high': 0.20,
        'confidence': 'medium',
        'description': 'Base added mass as fraction of boat mass',
        'source': 'Newman (1977) gives 5-15% for slender hulls in surge'
    },
    
    # Trim parameters
    'pitch_damping': {
        'baseline': 5000,
        'low': 1000,
        'high': 15000,
        'confidence': 'low',
        'description': 'Pitch damping coefficient (N·m·s/rad)',
        'source': 'Estimated - no direct measurement'
    },
    'weight_shift_magnitude': {
        'baseline': 0.15,
        'low': 0.05,
        'high': 0.30,
        'confidence': 'low',
        'description': 'Maximum crew weight shift (meters)',
        'source': 'Assumed based on paddling motion'
    },
}


# =============================================================================
# Sensitivity Analysis Functions
# =============================================================================

def run_single_simulation(params: Dict) -> Tuple[float, float]:
    """
    Run simulation with given parameters and return (sync_velocity, random_velocity).
    
    """
    sync_offsets = [0.0] * 20
    np.random.seed(42)
    random_offsets = list(np.random.uniform(-0.25, 0.25, 20))

    sync_result = simulate_enhanced(sync_offsets, **params)
    random_result = simulate_enhanced(random_offsets, **params)
    return sync_result.mean_velocity, random_result.mean_velocity


def one_at_a_time_sensitivity(n_points: int = 11) -> Dict:
    """
    One-at-a-time (OAT) sensitivity analysis.
    
    Vary each parameter independently while holding others at baseline.
    This identifies which parameters most strongly influence the result.
    
    Returns:
        Dictionary mapping parameter names to arrays of (param_value, sync_v, random_v, pct_diff)
    """
    results = {}
    
    for param_name, param_info in PARAMETER_RANGES.items():
        baseline = param_info['baseline']
        low = param_info['low']
        high = param_info['high']
        
        # Create parameter sweep
        values = np.linspace(low, high, n_points)
        
        sweep_results = []
        for val in values:
            # Create parameter dict with this value changed
            params = {k: v['baseline'] for k, v in PARAMETER_RANGES.items()}
            params[param_name] = val
            
            # Run simulation
            sync_v, random_v = run_single_simulation(params)
            pct_diff = (random_v - sync_v) / sync_v * 100
            
            sweep_results.append({
                'value': val,
                'sync_velocity': sync_v,
                'random_velocity': random_v,
                'pct_diff': pct_diff
            })
        
        results[param_name] = sweep_results
        print(f"Completed sweep for {param_name}")
    
    return results


def compute_sensitivity_indices(oat_results: Dict) -> Dict:
    """
    Compute sensitivity indices from OAT analysis.
    
    For each parameter, compute:
    - Range of pct_diff across parameter sweep
    - Normalized sensitivity (range / parameter range)
    - Direction (does increasing parameter favor sync or desync?)
    """
    indices = {}
    
    for param_name, sweep in oat_results.items():
        pct_diffs = [r['pct_diff'] for r in sweep]
        values = [r['value'] for r in sweep]
        
        diff_range = max(pct_diffs) - min(pct_diffs)
        param_range = max(values) - min(values)
        
        # Normalized sensitivity: how much does pct_diff change per unit parameter change
        if param_range > 0:
            normalized = diff_range / param_range
        else:
            normalized = 0
        
        # Direction: positive means increasing parameter favors desync
        if len(pct_diffs) > 1:
            direction = np.sign(pct_diffs[-1] - pct_diffs[0])
        else:
            direction = 0
        
        indices[param_name] = {
            'diff_range': diff_range,
            'normalized_sensitivity': normalized,
            'direction': direction,
            'confidence': PARAMETER_RANGES[param_name]['confidence']
        }
    
    return indices


def monte_carlo_sensitivity(n_samples: int = 1000) -> Dict:
    """
    Global sensitivity analysis using Monte Carlo sampling.
    
    Randomly sample parameters from their plausible ranges and run simulations.
    This captures interaction effects that OAT misses.
    
    Returns:
        Dictionary with sampled parameters and results
    """
    results = {
        'samples': [],
        'sync_velocities': [],
        'random_velocities': [],
        'pct_diffs': []
    }
    
    for i in range(n_samples):
        # Sample each parameter uniformly from its range
        params = {}
        for param_name, param_info in PARAMETER_RANGES.items():
            low = param_info['low']
            high = param_info['high']
            params[param_name] = np.random.uniform(low, high)
        
        # Run simulation
        sync_v, random_v = run_single_simulation(params)
        pct_diff = (random_v - sync_v) / sync_v * 100
        
        results['samples'].append(params)
        results['sync_velocities'].append(sync_v)
        results['random_velocities'].append(random_v)
        results['pct_diffs'].append(pct_diff)
        
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{n_samples} Monte Carlo samples")
    
    return results


def find_conclusion_flip_conditions(mc_results: Dict) -> Dict:
    """
    Identify parameter combinations where the conclusion flips.
    
    Find samples where:
    - pct_diff > 0: desync is faster (our main conclusion)
    - pct_diff < 0: sync is faster (conclusion flips)
    - |pct_diff| < 0.5%: effectively no difference
    """
    pct_diffs = np.array(mc_results['pct_diffs'])
    samples = mc_results['samples']
    
    desync_wins = pct_diffs > 0.5
    sync_wins = pct_diffs < -0.5
    tie = np.abs(pct_diffs) <= 0.5
    
    analysis = {
        'desync_wins_fraction': np.mean(desync_wins),
        'sync_wins_fraction': np.mean(sync_wins),
        'tie_fraction': np.mean(tie),
        'pct_diff_mean': np.mean(pct_diffs),
        'pct_diff_std': np.std(pct_diffs),
        'pct_diff_5th': np.percentile(pct_diffs, 5),
        'pct_diff_95th': np.percentile(pct_diffs, 95),
    }
    
    # Find parameter patterns that lead to sync winning
    if np.sum(sync_wins) > 0:
        sync_samples = [s for s, win in zip(samples, sync_wins) if win]
        analysis['sync_wins_conditions'] = {
            param: {
                'mean': np.mean([s[param] for s in sync_samples]),
                'std': np.std([s[param] for s in sync_samples])
            }
            for param in PARAMETER_RANGES.keys()
        }
    
    return analysis


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_oat_sensitivity(oat_results: Dict, save_path: str = None):
    """Create tornado plot and individual sensitivity curves."""
    
    n_params = len(oat_results)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top: Tornado plot showing sensitivity ranges
    ax1 = axes[0]
    
    param_names = list(oat_results.keys())
    ranges = []
    baselines = []
    
    for param in param_names:
        pct_diffs = [r['pct_diff'] for r in oat_results[param]]
        ranges.append((min(pct_diffs), max(pct_diffs)))
        # Find baseline result (middle of sweep)
        mid_idx = len(pct_diffs) // 2
        baselines.append(pct_diffs[mid_idx])
    
    # Sort by range size
    range_sizes = [r[1] - r[0] for r in ranges]
    sorted_idx = np.argsort(range_sizes)[::-1]
    
    y_pos = np.arange(n_params)
    for i, idx in enumerate(sorted_idx):
        low, high = ranges[idx]
        color = 'red' if PARAMETER_RANGES[param_names[idx]]['confidence'] == 'very_low' else \
                'orange' if PARAMETER_RANGES[param_names[idx]]['confidence'] == 'low' else 'green'
        ax1.barh(i, high - low, left=low, color=color, alpha=0.7, edgecolor='black')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([param_names[i] for i in sorted_idx])
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Velocity Difference: Random vs Sync (%)')
    ax1.set_title('Tornado Plot: Parameter Sensitivity\n(Red = Very Low Confidence, Orange = Low, Green = Medium)')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Bottom: Individual sensitivity curves for top 4 parameters
    ax2 = axes[1]
    
    top_params = [param_names[i] for i in sorted_idx[:4]]
    colors = plt.cm.viridis(np.linspace(0, 0.8, 4))
    
    for param, color in zip(top_params, colors):
        values = [r['value'] for r in oat_results[param]]
        pct_diffs = [r['pct_diff'] for r in oat_results[param]]
        
        # Normalize x-axis to [0, 1] for comparison
        values_norm = (np.array(values) - min(values)) / (max(values) - min(values))
        
        ax2.plot(values_norm, pct_diffs, 'o-', color=color, label=param, linewidth=2, markersize=6)
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Parameter Value (Normalized: 0 = Low, 1 = High)')
    ax2.set_ylabel('Velocity Difference (%)')
    ax2.set_title('Sensitivity Curves for Most Influential Parameters')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_monte_carlo_results(mc_results: Dict, save_path: str = None):
    """Visualize Monte Carlo sensitivity analysis results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    pct_diffs = np.array(mc_results['pct_diffs'])
    
    # Top left: Histogram of outcomes
    ax1 = axes[0, 0]
    ax1.hist(pct_diffs, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference')
    ax1.axvline(x=np.mean(pct_diffs), color='blue', linestyle='-', linewidth=2, 
                label=f'Mean: {np.mean(pct_diffs):.2f}%')
    ax1.axvline(x=np.percentile(pct_diffs, 5), color='gray', linestyle=':', linewidth=1)
    ax1.axvline(x=np.percentile(pct_diffs, 95), color='gray', linestyle=':', linewidth=1,
                label=f'90% CI: [{np.percentile(pct_diffs, 5):.1f}%, {np.percentile(pct_diffs, 95):.1f}%]')
    ax1.set_xlabel('Velocity Difference: Random vs Sync (%)')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Outcomes Across Parameter Space')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top right: Fraction of outcomes by category
    ax2 = axes[0, 1]
    desync_wins = np.mean(pct_diffs > 0.5) * 100
    sync_wins = np.mean(pct_diffs < -0.5) * 100
    tie = np.mean(np.abs(pct_diffs) <= 0.5) * 100
    
    categories = ['Desync Faster\n(>0.5%)', 'No Difference\n(±0.5%)', 'Sync Faster\n(<-0.5%)']
    values = [desync_wins, tie, sync_wins]
    colors = ['green', 'gray', 'red']
    
    bars = ax2.bar(categories, values, color=colors, edgecolor='black', alpha=0.7)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Fraction of Parameter Space (%)')
    ax2.set_title('How Often Does Each Conclusion Hold?')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Bottom: Scatter plots showing parameter correlations with outcome
    # Show the two most influential parameters
    samples = mc_results['samples']
    
    ax3 = axes[1, 0]
    coupling_strength = [s['coupling_strength'] for s in samples]
    ax3.scatter(coupling_strength, pct_diffs, alpha=0.3, s=10)
    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.set_xlabel('Coupling Strength')
    ax3.set_ylabel('Velocity Difference (%)')
    ax3.set_title('Effect of Coupling Strength Parameter')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    coupling_distance = [s['coupling_distance'] for s in samples]
    ax4.scatter(coupling_distance, pct_diffs, alpha=0.3, s=10)
    ax4.axhline(y=0, color='red', linestyle='--')
    ax4.set_xlabel('Coupling Distance (m)')
    ax4.set_ylabel('Velocity Difference (%)')
    ax4.set_title('Effect of Coupling Distance Parameter')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def generate_sensitivity_report(oat_results: Dict, mc_results: Dict) -> str:
    """Generate a text report summarizing sensitivity analysis findings."""
    
    sensitivity_indices = compute_sensitivity_indices(oat_results)
    flip_analysis = find_conclusion_flip_conditions(mc_results)
    
    report = []
    report.append("=" * 70)
    report.append("SENSITIVITY ANALYSIS REPORT")
    report.append("=" * 70)
    
    report.append("\n1. PARAMETER SENSITIVITY RANKING")
    report.append("-" * 40)
    
    # Sort by sensitivity
    sorted_params = sorted(sensitivity_indices.items(), 
                          key=lambda x: x[1]['diff_range'], reverse=True)
    
    for i, (param, info) in enumerate(sorted_params, 1):
        direction = "↑ desync" if info['direction'] > 0 else "↑ sync" if info['direction'] < 0 else "neutral"
        report.append(f"{i}. {param}")
        report.append(f"   Range of effect: {info['diff_range']:.2f}%")
        report.append(f"   Direction: Increasing parameter → {direction}")
        report.append(f"   Confidence: {info['confidence']}")
    
    report.append("\n2. MONTE CARLO RESULTS")
    report.append("-" * 40)
    report.append(f"Samples: {len(mc_results['pct_diffs'])}")
    report.append(f"Mean velocity difference: {flip_analysis['pct_diff_mean']:.2f}%")
    report.append(f"Std deviation: {flip_analysis['pct_diff_std']:.2f}%")
    report.append(f"90% Confidence interval: [{flip_analysis['pct_diff_5th']:.2f}%, {flip_analysis['pct_diff_95th']:.2f}%]")
    
    report.append("\n3. ROBUSTNESS OF CONCLUSIONS")
    report.append("-" * 40)
    report.append(f"Desync faster (>0.5%): {flip_analysis['desync_wins_fraction']*100:.1f}% of parameter space")
    report.append(f"No clear winner (±0.5%): {flip_analysis['tie_fraction']*100:.1f}% of parameter space")
    report.append(f"Sync faster (<-0.5%): {flip_analysis['sync_wins_fraction']*100:.1f}% of parameter space")
    
    return "\n".join(report)


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("Dragon Boat Synchronization: Sensitivity Analysis")
    print("=" * 60)
        
    print("\n1. Running One-at-a-Time sensitivity analysis...")
    print("   (Varying each parameter independently)")
    oat_results = one_at_a_time_sensitivity(n_points=11)
    
    print("\n2. Running Monte Carlo sensitivity analysis...")
    print("   (Random sampling from full parameter space)")
    mc_results = monte_carlo_sensitivity(n_samples=1000)
    
    print("\n3. Generating visualizations...")
    plot_oat_sensitivity(oat_results, "figures/sensitivity_tornado.png")
    plot_monte_carlo_results(mc_results, "figures/sensitivity_monte_carlo.png")
    
    print("\n4. Generating report...")
    report = generate_sensitivity_report(oat_results, mc_results)
    print(report)
    
    with open("figures/sensitivity_report.txt", "w") as f:
        f.write(report)