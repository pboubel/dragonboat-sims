"""
Dragon Boat Physics Simulation: Does Synchronization Affect Velocity?

Physical Model:
- Each paddler applies a force pulse during their stroke
- Hull drag is proportional to v^2 (standard hydrodynamic drag)
- Paddle drag during stroke is localized and independent per paddle
- We compare: perfectly synchronized vs. phase-offset paddlers

Author: Paula Boubel (paulaboubel@gmail.com)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import json

# =============================================================================
# Physical Constants and Boat Parameters
# =============================================================================

@dataclass
class BoatParams:
    """Dragon boat physical parameters"""
    mass: float = 1200.0          # kg (boat + 20 paddlers + drummer + steersperson)
    length: float = 12.5          # m (standard dragon boat)
    beam: float = 1.12            # m (width)
    wetted_area: float = 15.0     # m^2 (approximate)
    
    # Hydrodynamic drag coefficient (hull)
    # F_drag = 0.5 * rho * Cd * A * v^2
    Cd_hull: float = 0.01         # Drag coefficient for slender hull
    rho_water: float = 1000.0     # kg/m^3
    
    @property
    def hull_drag_coeff(self) -> float:
        """Combined coefficient: 0.5 * rho * Cd * A"""
        return 0.5 * self.rho_water * self.Cd_hull * self.wetted_area


@dataclass 
class PaddleParams:
    """Individual paddle stroke parameters"""
    # Force profile: modeled as half-sine pulse
    peak_force: float = 250.0     # N per paddler at peak
    stroke_duration: float = 0.5  # seconds (catch to exit)
    stroke_rate: float = 60.0     # strokes per minute
    
    # Paddle drag (resistance during recovery)
    paddle_drag_coeff: float = 5.0  # Small drag during paddle movement
    
    @property
    def stroke_period(self) -> float:
        """Time between stroke starts"""
        return 60.0 / self.stroke_rate


# =============================================================================
# Force Model
# =============================================================================

def stroke_force_profile(t: float, stroke_start: float, params: PaddleParams) -> float:
    """
    Calculate instantaneous force from a single paddle stroke.
    
    Uses a half-sine profile: F(t) = F_peak * sin(pi * (t - t_start) / duration)
    This is more realistic than a square pulse.
    
    Args:
        t: Current time (s)
        stroke_start: When this stroke began (s)
        params: Paddle parameters
    
    Returns:
        Force in Newtons (0 if outside stroke window)
    """
    t_rel = t - stroke_start
    
    if 0 <= t_rel <= params.stroke_duration:
        # Half-sine profile during active stroke
        phase = np.pi * t_rel / params.stroke_duration
        return params.peak_force * np.sin(phase)
    else:
        return 0.0


def get_stroke_starts(t: float, phase_offset: float, params: PaddleParams) -> List[float]:
    """
    Get all stroke start times that could affect time t.
    
    Args:
        t: Current simulation time
        phase_offset: This paddler's phase offset (0 to 1, fraction of period)
        params: Paddle parameters
    
    Returns:
        List of stroke start times
    """
    period = params.stroke_period
    offset_time = phase_offset * period
    
    # Find the stroke number we're in
    stroke_num = int((t - offset_time) / period)
    
    # Check this stroke and adjacent ones
    starts = []
    for n in [stroke_num - 1, stroke_num, stroke_num + 1]:
        start = n * period + offset_time
        if start >= 0:
            starts.append(start)
    
    return starts


def paddler_force(t: float, phase_offset: float, params: PaddleParams) -> float:
    """
    Total force from one paddler at time t.
    
    Args:
        t: Current time
        phase_offset: Phase offset (0 = synchronized with reference)
        params: Paddle parameters
    
    Returns:
        Force in Newtons
    """
    stroke_starts = get_stroke_starts(t, phase_offset, params)
    total_force = sum(stroke_force_profile(t, start, params) for start in stroke_starts)
    return total_force


# =============================================================================
# Drag Models
# =============================================================================

def hull_drag(velocity: float, boat: BoatParams) -> float:
    """
    Hydrodynamic drag on the hull.
    
    F_drag = 0.5 * rho * Cd * A * v^2
    
    """
    return boat.hull_drag_coeff * velocity * abs(velocity)


def paddle_splash_drag(velocity: float, n_paddlers_active: int, paddle: PaddleParams) -> float:
    """
    Additional drag from paddles in water.
    
    This is small and proportional to how many paddles are currently
    in the water. Importantly: this doesn't depend on synchronization,
    only on the COUNT of active paddles at any instant.
    """
    return paddle.paddle_drag_coeff * n_paddlers_active * velocity


# =============================================================================
# Simulation Engine
# =============================================================================

@dataclass
class SimulationResult:
    """Container for simulation outputs"""
    time: np.ndarray
    velocity: np.ndarray
    position: np.ndarray
    total_force: np.ndarray
    drag_force: np.ndarray
    mean_velocity: float
    final_velocity: float
    distance_traveled: float
    phase_offsets: List[float]
    label: str


def simulate_dragon_boat(
    phase_offsets: List[float],
    duration: float = 30.0,
    dt: float = 0.001,
    boat: BoatParams = None,
    paddle: PaddleParams = None,
    label: str = "Simulation"
) -> SimulationResult:
    """
    Simulate dragon boat motion with given paddler phase offsets.
    
    Args:
        phase_offsets: List of phase offsets for each paddler (0-1)
        duration: Simulation duration (s)
        dt: Time step (s)
        boat: Boat parameters
        paddle: Paddle parameters
        label: Label for this simulation
    
    Returns:
        SimulationResult with all time series data
    """
    if boat is None:
        boat = BoatParams()
    if paddle is None:
        paddle = PaddleParams()
    
    n_paddlers = len(phase_offsets)
    n_steps = int(duration / dt)
    
    # Arrays for results
    time = np.zeros(n_steps)
    velocity = np.zeros(n_steps)
    position = np.zeros(n_steps)
    total_force = np.zeros(n_steps)
    drag_force = np.zeros(n_steps)
    
    # Initial conditions
    v = 0.0
    x = 0.0
    
    for i in range(n_steps):
        t = i * dt
        time[i] = t
        
        # Calculate propulsive force from all paddlers
        prop_force = sum(
            paddler_force(t, offset, paddle) 
            for offset in phase_offsets
        )
        
        # Count active paddlers (for splash drag)
        n_active = sum(
            1 for offset in phase_offsets
            for start in get_stroke_starts(t, offset, paddle)
            if 0 <= (t - start) <= paddle.stroke_duration
        )
        
        # Calculate drag
        f_hull_drag = hull_drag(v, boat)
        f_paddle_drag = paddle_splash_drag(v, n_active, paddle)
        f_drag = f_hull_drag + f_paddle_drag
        
        # Net force
        f_net = prop_force - f_drag
        
        # Store values
        total_force[i] = prop_force
        drag_force[i] = f_drag
        velocity[i] = v
        position[i] = x
        
        # Euler integration (sufficient for this dt)
        a = f_net / boat.mass
        v += a * dt
        x += v * dt
        
        # Velocity can't go negative (no reverse paddling modeled)
        v = max(0, v)
    
    # Calculate summary statistics (skip startup transient)
    steady_start = int(10.0 / dt)  # Start measuring after 10s
    mean_v = np.mean(velocity[steady_start:])
    
    return SimulationResult(
        time=time,
        velocity=velocity,
        position=position,
        total_force=total_force,
        drag_force=drag_force,
        mean_velocity=mean_v,
        final_velocity=velocity[-1],
        distance_traveled=position[-1],
        phase_offsets=phase_offsets,
        label=label
    )


# =============================================================================
# Experiment Configurations
# =============================================================================

def create_synchronized_offsets(n_paddlers: int = 20) -> List[float]:
    """All paddlers perfectly synchronized"""
    return [0.0] * n_paddlers


def create_random_offsets(n_paddlers: int = 20, max_offset: float = 0.2, seed: int = 42) -> List[float]:
    """
    Random phase offsets within a window.
    
    max_offset: Maximum offset as fraction of stroke period
                0.2 = +/-20% of period = +/-0.2s at 60 spm
    """
    np.random.seed(seed)
    return list(np.random.uniform(-max_offset, max_offset, n_paddlers))


def create_alternating_offsets(n_paddlers: int = 20, offset: float = 0.1) -> List[float]:
    """Left side vs right side slightly offset"""
    offsets = []
    for i in range(n_paddlers):
        if i % 2 == 0:  # Left side
            offsets.append(-offset / 2)
        else:  # Right side
            offsets.append(offset / 2)
    return offsets


def create_wave_offsets(n_paddlers: int = 20, max_offset: float = 0.15) -> List[float]:
    """
    Front-to-back wave pattern (like sometimes happens naturally).
    Each row slightly behind the one in front.
    """
    offsets = []
    n_rows = n_paddlers // 2
    for i in range(n_paddlers):
        row = i // 2
        offset = (row / n_rows) * max_offset
        offsets.append(offset)
    return offsets


# =============================================================================
# Analysis and Visualization
# =============================================================================

def run_comparison_experiment() -> dict:
    """
    Run the main experiment comparing synchronized vs unsynchronized crews.
    """
    
    experiments = [
        ("Perfect Sync", create_synchronized_offsets()),
        ("Random +/-10%", create_random_offsets(max_offset=0.1, seed=42)),
        ("Random +/-20%", create_random_offsets(max_offset=0.2, seed=42)),
        ("Random +/-30%", create_random_offsets(max_offset=0.3, seed=42)),
        ("Left/Right Split", create_alternating_offsets(offset=0.15)),
        ("Front-to-Back Wave", create_wave_offsets(max_offset=0.2)),
    ]
    
    results = []
    for label, offsets in experiments:
        print(f"Running: {label}...")
        result = simulate_dragon_boat(offsets, duration=30.0, label=label)
        results.append(result)
        print(f"  Mean velocity: {result.mean_velocity:.4f} m/s")
        print(f"  Distance: {result.distance_traveled:.2f} m")
    
    return results


def plot_results(results: List[SimulationResult], save_path: str = None):
    """Create visualization of simulation results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
    
    # Plot 1: Velocity over time
    ax1 = axes[0, 0]
    for result, color in zip(results, colors):
        ax1.plot(result.time, result.velocity, label=result.label, color=color, alpha=0.8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.set_title("Boat Velocity Over Time")
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Position over time
    ax2 = axes[0, 1]
    for result, color in zip(results, colors):
        ax2.plot(result.time, result.position, label=result.label, color=color, alpha=0.8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position (m)")
    ax2.set_title("Distance Traveled")
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Zoomed velocity (steady state)
    ax3 = axes[1, 0]
    for result, color in zip(results, colors):
        mask = result.time > 15  # Show only steady state
        ax3.plot(result.time[mask], result.velocity[mask], label=result.label, color=color, alpha=0.8)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Velocity (m/s)")
    ax3.set_title("Steady-State Velocity (Zoomed)")
    ax3.legend(loc='lower right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary bar chart
    ax4 = axes[1, 1]
    labels = [r.label for r in results]
    mean_velocities = [r.mean_velocity for r in results]
    
    bars = ax4.bar(range(len(labels)), mean_velocities, color=colors)
    ax4.set_xticks(range(len(labels)))
    ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel("Mean Velocity (m/s)")
    ax4.set_title("Mean Steady-State Velocity Comparison")
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, v in zip(bars, mean_velocities):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add reference line at synchronized value
    ax4.axhline(y=mean_velocities[0], color='red', linestyle='--', alpha=0.5, 
                label=f'Sync baseline: {mean_velocities[0]:.3f}')
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    return fig


def analyze_velocity_differences(results: List[SimulationResult]) -> str:
    """Generate analysis report"""
    
    sync_result = results[0]
    
    report = []
    report.append("\n" + "=" * 70)
    report.append("ANALYSIS REPORT")
    report.append("=" * 70)
    
    report.append(f"\nBaseline (Perfect Sync): {sync_result.mean_velocity:.4f} m/s")
    report.append("\nComparison with other timing patterns:")
    report.append("-" * 50)
    
    for result in results[1:]:
        diff = result.mean_velocity - sync_result.mean_velocity
        pct_diff = (diff / sync_result.mean_velocity) * 100
        report.append(f"\n{result.label}:")
        report.append(f"  Mean velocity: {result.mean_velocity:.4f} m/s")
        report.append(f"  Difference: {diff:+.4f} m/s ({pct_diff:+.3f}%)")
    
    # Statistical summary
    velocities = [r.mean_velocity for r in results]
    max_diff = max(velocities) - min(velocities)
    pct_spread = (max_diff / sync_result.mean_velocity) * 100
    
    report.append("\n" + "-" * 50)
    report.append(f"\nTotal velocity spread: {max_diff:.4f} m/s ({pct_spread:.3f}%)")
    report.append(f"Mean of all configurations: {np.mean(velocities):.4f} m/s")
    report.append(f"Std deviation: {np.std(velocities):.4f} m/s")

    return "\n".join(report)


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Run the experiment
    results = run_comparison_experiment()
    
    # Generate plots
    fig = plot_results(results, save_path="figures/dragon_boat_results.png")

    # Print analysis
    analysis = analyze_velocity_differences(results)
    print(analysis)
    
    # Save full report
    with open("figures/dragon_boat_analysis.txt", "w") as f:
        f.write("DRAGON BOAT SYNCHRONIZATION SIMULATION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("SIMULATION PARAMETERS\n")
        f.write("-" * 30 + "\n")
        boat = BoatParams()
        paddle = PaddleParams()
        f.write(f"Boat mass: {boat.mass} kg\n")
        f.write(f"Hull drag coefficient: {boat.Cd_hull}\n")
        f.write(f"Paddlers: 20\n")
        f.write(f"Peak force per paddler: {paddle.peak_force} N\n")
        f.write(f"Stroke rate: {paddle.stroke_rate} spm\n")
        f.write(f"Stroke duration: {paddle.stroke_duration} s\n\n")
        
        f.write("RESULTS SUMMARY\n")
        f.write("-" * 30 + "\n")
        for r in results:
            f.write(f"{r.label}:\n")
            f.write(f"  Mean velocity: {r.mean_velocity:.4f} m/s\n")
            f.write(f"  Final velocity: {r.final_velocity:.4f} m/s\n")
            f.write(f"  Distance: {r.distance_traveled:.2f} m\n\n")
        
        f.write(analysis)
        f.write("\n\n")