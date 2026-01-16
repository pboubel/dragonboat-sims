"""
Dragon Boat Physics: Quantifying Second-Order Effects

This simulation adds the four previously-neglected effects:
1. Wake interactions between paddles
2. Blade-to-blade hydrodynamic coupling  
3. Added mass effects during acceleration
4. Crew weight shifts affecting hull trim

We'll quantify each effect's contribution to see if synchronization
matters once these are included.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from scipy.integrate import solve_ivp
import warnings

# =============================================================================
# Enhanced Physical Parameters
# =============================================================================

@dataclass
class EnhancedBoatParams:
    """Dragon boat parameters with second-order effects"""
    # Basic parameters
    mass: float = 1200.0              # kg (boat + crew)
    length: float = 12.5              # m
    beam: float = 1.12                # m
    draft: float = 0.15               # m (how deep hull sits)
    wetted_area: float = 15.0         # m²
    
    # Drag parameters
    Cd_base: float = 0.010            # Base drag coefficient
    rho_water: float = 1000.0         # kg/m³
    
    # Added mass (typically 5-15% for slender hulls)
    added_mass_coeff: float = 0.10    # Added mass as fraction of boat mass
    
    # Trim sensitivity
    # Change in Cd per degree of trim angle
    Cd_trim_sensitivity: float = 0.001  # ΔCd per degree
    
    # Longitudinal center of gravity from stern (m)
    lcg_base: float = 6.25            # Centered by default
    
    # Moment of inertia for pitching (rough estimate)
    pitch_inertia: float = 15000.0    # kg·m² 
    
    # Pitch damping coefficient
    pitch_damping: float = 5000.0     # N·m·s/rad
    
    # Pitch restoring force (hydrostatic stiffness)
    pitch_stiffness: float = 50000.0  # N·m/rad
    
    @property
    def effective_mass(self) -> float:
        """Mass including added mass"""
        return self.mass * (1 + self.added_mass_coeff)
    
    @property
    def hull_drag_coeff(self) -> float:
        """Base drag coefficient: 0.5 * rho * Cd * A"""
        return 0.5 * self.rho_water * self.Cd_base * self.wetted_area


@dataclass
class EnhancedPaddleParams:
    """Paddle parameters with position information"""
    peak_force: float = 250.0         # N per paddler
    stroke_duration: float = 0.5      # seconds
    stroke_rate: float = 60.0         # strokes per minute
    
    # Paddle geometry
    blade_area: float = 0.06          # m² (typical dragon boat paddle)
    blade_Cd: float = 1.2             # Drag coefficient of blade
    shaft_length: float = 1.2         # m
    
    # Wake parameters
    wake_decay_time: float = 0.3      # seconds for wake to dissipate
    wake_decay_distance: float = 0.5  # meters for spatial decay
    wake_strength: float = 0.15       # fraction of stroke force affected
    
    # Coupling parameters  
    coupling_distance: float = 1.2    # m (distance for significant coupling)
    coupling_strength: float = 0.05   # max efficiency reduction from coupling
    
    @property
    def stroke_period(self) -> float:
        return 60.0 / self.stroke_rate


@dataclass
class PaddlerPosition:
    """Position of a single paddler"""
    x: float              # Distance from stern (m)
    side: str             # 'port' or 'starboard'
    phase_offset: float   # Timing offset (fraction of period)
    
    # Paddler mass for weight shift calculations
    mass: float = 75.0    # kg


def create_crew_positions(n_paddlers: int = 20, phase_offsets: List[float] = None) -> List[PaddlerPosition]:
    """
    Create realistic crew positions for a dragon boat.
    
    Standard layout: 10 rows of 2 paddlers (port/starboard)
    Spacing: ~0.9m between rows
    First row starts ~2m from bow
    """
    if phase_offsets is None:
        phase_offsets = [0.0] * n_paddlers
    
    positions = []
    n_rows = n_paddlers // 2
    row_spacing = 0.9  # meters between rows
    first_row_from_bow = 2.0  # meters
    boat_length = 12.5
    
    for i in range(n_paddlers):
        row = i // 2
        side = 'port' if i % 2 == 0 else 'starboard'
        
        # X position (from stern)
        x = boat_length - first_row_from_bow - row * row_spacing
        
        positions.append(PaddlerPosition(
            x=x,
            side=side,
            phase_offset=phase_offsets[i]
        ))
    
    return positions


# =============================================================================
# Effect 1: Wake Interactions
# =============================================================================

class WakeField:
    """
    Models the wake left behind by paddle strokes.
    
    Each stroke creates a wake that:
    - Decays exponentially in time
    - Decays with distance from the stroke location
    - Reduces effectiveness of subsequent strokes in the same region
    """
    
    def __init__(self, params: EnhancedPaddleParams):
        self.params = params
        self.wake_events = []  # List of (time, x_position, side, strength)
    
    def add_stroke(self, t: float, x: float, side: str, strength: float = 1.0):
        """Record a paddle stroke that creates wake"""
        self.wake_events.append((t, x, side, strength))
        
        # Prune old events (older than 3 decay times)
        cutoff = t - 3 * self.params.wake_decay_time
        self.wake_events = [(te, xe, se, st) for te, xe, se, st in self.wake_events if te > cutoff]
    
    def get_wake_penalty(self, t: float, x: float, side: str) -> float:
        """
        Calculate efficiency reduction due to wake from previous strokes.
        
        Returns: Fraction reduction in stroke effectiveness (0 to wake_strength)
        """
        total_penalty = 0.0
        
        for t_event, x_event, side_event, strength in self.wake_events:
            # Only same-side wakes matter significantly
            if side_event != side:
                continue
            
            # Time decay
            dt = t - t_event
            if dt <= 0:
                continue
            time_factor = np.exp(-dt / self.params.wake_decay_time)
            
            # Spatial decay
            dx = abs(x - x_event)
            space_factor = np.exp(-dx / self.params.wake_decay_distance)
            
            # Combine
            penalty = self.params.wake_strength * strength * time_factor * space_factor
            total_penalty += penalty
        
        # Cap at maximum wake strength
        return min(total_penalty, self.params.wake_strength)


# =============================================================================
# Effect 2: Blade-to-Blade Hydrodynamic Coupling
# =============================================================================

def calculate_coupling_penalty(
    positions: List[PaddlerPosition],
    active_mask: np.ndarray,
    params: EnhancedPaddleParams
) -> np.ndarray:
    """
    Calculate efficiency reduction due to nearby paddles operating simultaneously.
    
    When two paddles are in the water at the same time and close together,
    they "compete" for water, reducing each other's effectiveness.
    
    Args:
        positions: List of paddler positions
        active_mask: Boolean array of which paddlers are currently stroking
        params: Paddle parameters
    
    Returns:
        Array of efficiency multipliers (0.95 to 1.0 typically)
    """
    n = len(positions)
    efficiency = np.ones(n)
    
    for i in range(n):
        if not active_mask[i]:
            continue
            
        for j in range(n):
            if i == j or not active_mask[j]:
                continue
            
            # Same side paddles have stronger coupling
            same_side = positions[i].side == positions[j].side
            
            # Distance between paddlers
            dx = abs(positions[i].x - positions[j].x)
            
            # Coupling strength decays with distance
            if same_side:
                # Same side: stronger coupling, closer spacing
                coupling = params.coupling_strength * np.exp(-dx / params.coupling_distance)
            else:
                # Opposite sides: weaker coupling (factor of 0.3)
                coupling = 0.3 * params.coupling_strength * np.exp(-dx / (2 * params.coupling_distance))
            
            efficiency[i] -= coupling
    
    # Ensure efficiency doesn't go below reasonable minimum
    return np.maximum(efficiency, 0.85)


# =============================================================================
# Effect 3: Added Mass During Acceleration
# =============================================================================

def calculate_added_mass_effect(
    velocity: float,
    acceleration: float,
    boat: EnhancedBoatParams
) -> float:
    """
    Calculate the effective mass including added mass effects.
    
    Added mass is most significant during acceleration phases.
    The effect scales with acceleration magnitude.
    
    Returns: Effective mass multiplier
    """
    # Added mass coefficient increases with acceleration
    # (This is a simplification - true added mass is complex)
    base_added = boat.added_mass_coeff
    
    return boat.mass * (1 + base_added)


# =============================================================================
# Effect 4: Crew Weight Shifts and Hull Trim
# =============================================================================

@dataclass
class PitchState:
    """State of the boat's pitching motion"""
    angle: float = 0.0        # Pitch angle (radians, positive = bow up)
    angular_velocity: float = 0.0


def calculate_crew_moment(
    positions: List[PaddlerPosition],
    stroke_phases: np.ndarray,
    boat: EnhancedBoatParams
) -> float:
    """
    Calculate the pitching moment from crew weight shifts.
    
    During a stroke:
    - Catch (phase 0-0.3): Paddler leans forward (shifts weight toward bow)
    - Pull (phase 0.3-0.8): Paddler leans back (shifts weight toward stern)  
    - Recovery (phase 0.8-1.0): Return to neutral
    
    Args:
        positions: Paddler positions
        stroke_phases: Current phase (0-1) of each paddler's stroke cycle
        boat: Boat parameters
    
    Returns:
        Net pitching moment (N·m, positive = bow up)
    """
    total_moment = 0.0
    lcg = boat.lcg_base  # Neutral center of gravity
    
    for pos, phase in zip(positions, stroke_phases):
        # Weight shift pattern (meters from neutral position)
        # Positive = toward bow
        if phase < 0:
            # Not in active stroke
            shift = 0.0
        elif phase < 0.3:
            # Catch: lean forward
            shift = 0.15 * (phase / 0.3)  # Up to 15cm forward
        elif phase < 0.8:
            # Pull: lean back
            shift = 0.15 - 0.35 * ((phase - 0.3) / 0.5)  # From +15cm to -20cm
        else:
            # Recovery: return to neutral
            shift = -0.20 * (1 - (phase - 0.8) / 0.2)
        
        # Moment = mass * g * lever_arm
        # Lever arm = shift distance (weight moves relative to CG)
        moment = pos.mass * 9.81 * shift
        total_moment += moment
    
    return total_moment


def calculate_trim_drag_modifier(pitch_angle: float, boat: EnhancedBoatParams) -> float:
    """
    Calculate drag coefficient modifier based on trim angle.
    
    Optimal trim is approximately level (0 degrees).
    Both bow-up and bow-down increase drag.
    
    Returns: Multiplier on base drag coefficient
    """
    # Convert to degrees for intuition
    angle_deg = np.degrees(pitch_angle)
    
    # Quadratic increase in drag with trim angle
    # At ±2 degrees, drag increases by ~Cd_trim_sensitivity * 4
    drag_increase = boat.Cd_trim_sensitivity * angle_deg**2
    
    return 1.0 + drag_increase


# =============================================================================
# Enhanced Simulation Engine
# =============================================================================

@dataclass
class EnhancedSimulationResult:
    """Results from enhanced simulation"""
    time: np.ndarray
    velocity: np.ndarray
    position: np.ndarray
    pitch_angle: np.ndarray
    
    # Force components
    propulsive_force: np.ndarray
    drag_force: np.ndarray
    
    # Effect contributions
    wake_penalty_avg: float
    coupling_penalty_avg: float
    added_mass_effect_avg: float
    trim_drag_increase_avg: float
    
    # Summary statistics
    mean_velocity: float
    label: str


def get_stroke_phase(t: float, phase_offset: float, params: EnhancedPaddleParams) -> float:
    """
    Get the current phase of a paddler's stroke cycle.
    
    Returns: Phase from 0 to 1 (0 = start of stroke, 1 = end of cycle)
             Returns -1 if in recovery (not actively stroking)
    """
    period = params.stroke_period
    offset_time = phase_offset * period
    
    # Time within current cycle
    cycle_time = (t - offset_time) % period
    
    # Stroke occupies first stroke_duration of the period
    if cycle_time <= params.stroke_duration:
        return cycle_time / params.stroke_duration
    else:
        return -1  # Recovery phase


def stroke_force(phase: float, params: EnhancedPaddleParams) -> float:
    """Force output based on stroke phase (0-1)"""
    if phase < 0:
        return 0.0
    return params.peak_force * np.sin(np.pi * phase)


def simulate_enhanced(
    phase_offsets: List[float],
    duration: float = 30.0,
    dt: float = 0.002,
    boat: EnhancedBoatParams = None,
    paddle: EnhancedPaddleParams = None,
    include_wake: bool = True,
    include_coupling: bool = True,
    include_added_mass: bool = True,
    include_trim: bool = True,
    label: str = "Simulation",
    # Sensitivity analysis parameters
    wake_decay_time: float = None,
    wake_decay_distance: float = None,
    wake_max_penalty: float = None,  # Maps to wake_strength
    coupling_strength: float = None,
    coupling_distance: float = None,
    coupling_opposite_ratio: float = None,
    added_mass_coeff: float = None,
    pitch_damping: float = None,
    weight_shift_magnitude: float = None,
) -> EnhancedSimulationResult:
    """
    Run enhanced simulation with all second-order effects.
    """
    if boat is None:
        boat = EnhancedBoatParams()
    if paddle is None:
        paddle = EnhancedPaddleParams()
    
    # Apply sensitivity parameter overrides to boat
    if added_mass_coeff is not None:
        boat.added_mass_coeff = added_mass_coeff
    if pitch_damping is not None:
        boat.pitch_damping = pitch_damping
    
    # Apply sensitivity parameter overrides to paddle
    if wake_decay_time is not None:
        paddle.wake_decay_time = wake_decay_time
    if wake_decay_distance is not None:
        paddle.wake_decay_distance = wake_decay_distance
    if wake_max_penalty is not None:
        paddle.wake_strength = wake_max_penalty
    if coupling_strength is not None:
        paddle.coupling_strength = coupling_strength
    if coupling_distance is not None:
        paddle.coupling_distance = coupling_distance
    
    # Store coupling_opposite_ratio for use in coupling calculation
    _coupling_opposite_ratio = coupling_opposite_ratio if coupling_opposite_ratio is not None else 0.3
    
    # Store weight_shift_magnitude for use in crew moment calculation
    _weight_shift_magnitude = weight_shift_magnitude if weight_shift_magnitude is not None else 0.15
    
    # Create crew positions
    positions = create_crew_positions(len(phase_offsets), phase_offsets)
    n_paddlers = len(positions)
    
    # Initialize wake field
    wake_field = WakeField(paddle)
    
    # Time stepping
    n_steps = int(duration / dt)
    time = np.zeros(n_steps)
    velocity = np.zeros(n_steps)
    position = np.zeros(n_steps)
    pitch_angle = np.zeros(n_steps)
    pitch_velocity = np.zeros(n_steps)
    propulsive_force = np.zeros(n_steps)
    drag_force = np.zeros(n_steps)
    
    # Track effect magnitudes
    wake_penalties = []
    coupling_penalties = []
    added_mass_effects = []
    trim_effects = []
    
    # Initial state
    v = 0.0
    x = 0.0
    theta = 0.0  # Pitch angle
    omega = 0.0  # Pitch angular velocity
    
    # Track when paddlers last stroked (for wake)
    last_stroke_start = [-np.inf] * n_paddlers
    
    for i in range(n_steps):
        t = i * dt
        time[i] = t
        
        # Get stroke phases for all paddlers
        phases = np.array([get_stroke_phase(t, pos.phase_offset, paddle) for pos in positions])
        active_mask = phases >= 0
        
        # Track stroke starts for wake
        for j, (phase, pos) in enumerate(zip(phases, positions)):
            if phase >= 0 and phase < dt / paddle.stroke_duration:
                # Just started a stroke
                if t - last_stroke_start[j] > paddle.stroke_period * 0.5:
                    wake_field.add_stroke(t, pos.x, pos.side)
                    last_stroke_start[j] = t
        
        # Calculate base forces
        base_forces = np.array([stroke_force(p, paddle) for p in phases])
        
        # Apply Effect 1: Wake penalties
        if include_wake:
            wake_mods = np.array([
                1.0 - wake_field.get_wake_penalty(t, pos.x, pos.side)
                for pos in positions
            ])
            wake_penalties.append(1.0 - np.mean(wake_mods[active_mask]) if active_mask.any() else 0.0)
        else:
            wake_mods = np.ones(n_paddlers)
        
        # Apply Effect 2: Coupling penalties (with configurable opposite ratio)
        if include_coupling:
            coupling_mods = _calculate_coupling_penalty_with_ratio(
                positions, active_mask, paddle, _coupling_opposite_ratio
            )
            coupling_penalties.append(1.0 - np.mean(coupling_mods[active_mask]) if active_mask.any() else 0.0)
        else:
            coupling_mods = np.ones(n_paddlers)
        
        # Total propulsive force
        total_prop_force = np.sum(base_forces * wake_mods * coupling_mods)
        propulsive_force[i] = total_prop_force
        
        # Effect 4: Trim-modified drag
        if include_trim:
            trim_modifier = calculate_trim_drag_modifier(theta, boat)
            trim_effects.append(trim_modifier - 1.0)
        else:
            trim_modifier = 1.0
        
        # Calculate drag
        f_drag = boat.hull_drag_coeff * trim_modifier * v * abs(v)
        drag_force[i] = f_drag
        
        # Net force
        f_net = total_prop_force - f_drag
        
        # Effect 3: Added mass
        if include_added_mass:
            # Estimate acceleration for added mass calculation
            est_accel = f_net / boat.mass
            effective_mass = calculate_added_mass_effect(v, est_accel, boat)
            added_mass_effects.append(effective_mass / boat.mass - 1.0)
        else:
            effective_mass = boat.mass
        
        # Update velocity
        a = f_net / effective_mass
        v += a * dt
        v = max(0, v)
        x += v * dt
        
        # Update pitch dynamics (Effect 4)
        if include_trim:
            crew_moment = _calculate_crew_moment_with_shift(
                positions, phases, boat, _weight_shift_magnitude
            )
            # Pitch equation: I * d²θ/dt² = M_crew - c * dθ/dt - k * θ
            angular_accel = (crew_moment - boat.pitch_damping * omega - boat.pitch_stiffness * theta) / boat.pitch_inertia
            omega += angular_accel * dt
            theta += omega * dt
        
        # Store state
        velocity[i] = v
        position[i] = x
        pitch_angle[i] = theta
        pitch_velocity[i] = omega
    
    # Calculate statistics (skip first 10s for steady state)
    steady_start = int(10.0 / dt)
    mean_v = np.mean(velocity[steady_start:])
    
    return EnhancedSimulationResult(
        time=time,
        velocity=velocity,
        position=position,
        pitch_angle=pitch_angle,
        propulsive_force=propulsive_force,
        drag_force=drag_force,
        wake_penalty_avg=np.mean(wake_penalties) if wake_penalties else 0.0,
        coupling_penalty_avg=np.mean(coupling_penalties) if coupling_penalties else 0.0,
        added_mass_effect_avg=np.mean(added_mass_effects) if added_mass_effects else 0.0,
        trim_drag_increase_avg=np.mean(trim_effects) if trim_effects else 0.0,
        mean_velocity=mean_v,
        label=label
    )


def _calculate_coupling_penalty_with_ratio(
    positions: List[PaddlerPosition],
    active_mask: np.ndarray,
    params: EnhancedPaddleParams,
    opposite_ratio: float
) -> np.ndarray:
    """
    Calculate coupling penalty with configurable opposite-side ratio.
    """
    n = len(positions)
    efficiency = np.ones(n)
    
    for i in range(n):
        if not active_mask[i]:
            continue
            
        for j in range(n):
            if i == j or not active_mask[j]:
                continue
            
            same_side = positions[i].side == positions[j].side
            dx = abs(positions[i].x - positions[j].x)
            
            if same_side:
                coupling = params.coupling_strength * np.exp(-dx / params.coupling_distance)
            else:
                coupling = opposite_ratio * params.coupling_strength * np.exp(-dx / (2 * params.coupling_distance))
            
            efficiency[i] -= coupling
    
    return np.maximum(efficiency, 0.85)


def _calculate_crew_moment_with_shift(
    positions: List[PaddlerPosition],
    stroke_phases: np.ndarray,
    boat: EnhancedBoatParams,
    weight_shift_magnitude: float
) -> float:
    """
    Calculate crew moment with configurable weight shift magnitude.
    """
    total_moment = 0.0
    
    for pos, phase in zip(positions, stroke_phases):
        if phase < 0:
            shift = 0.0
        elif phase < 0.3:
            shift = weight_shift_magnitude * (phase / 0.3)
        elif phase < 0.8:
            shift = weight_shift_magnitude - (weight_shift_magnitude + weight_shift_magnitude * 0.33) * ((phase - 0.3) / 0.5)
        else:
            shift = -weight_shift_magnitude * 0.33 * (1 - (phase - 0.8) / 0.2)
        
        moment = pos.mass * 9.81 * shift
        total_moment += moment
    
    return total_moment


# =============================================================================
# Experiments
# =============================================================================

def run_effect_isolation_experiment():
    """
    Run experiments isolating each effect to quantify its contribution.
    """
    print("=" * 70)
    print("EFFECT ISOLATION EXPERIMENT")
    print("=" * 70)
    
    # Test configurations
    sync_offsets = [0.0] * 20
    np.random.seed(42)
    random_offsets = list(np.random.uniform(-0.25, 0.25, 20))
    
    results = {}
    
    # Baseline: no effects
    print("\n1. Baseline (no second-order effects)...")
    results['baseline_sync'] = simulate_enhanced(
        sync_offsets, include_wake=False, include_coupling=False,
        include_added_mass=False, include_trim=False, label="Baseline Sync"
    )
    results['baseline_random'] = simulate_enhanced(
        random_offsets, include_wake=False, include_coupling=False,
        include_added_mass=False, include_trim=False, label="Baseline Random"
    )
    
    # Effect 1: Wake only
    print("2. Wake interactions only...")
    results['wake_sync'] = simulate_enhanced(
        sync_offsets, include_wake=True, include_coupling=False,
        include_added_mass=False, include_trim=False, label="Wake Sync"
    )
    results['wake_random'] = simulate_enhanced(
        random_offsets, include_wake=True, include_coupling=False,
        include_added_mass=False, include_trim=False, label="Wake Random"
    )
    
    # Effect 2: Coupling only
    print("3. Blade coupling only...")
    results['coupling_sync'] = simulate_enhanced(
        sync_offsets, include_wake=False, include_coupling=True,
        include_added_mass=False, include_trim=False, label="Coupling Sync"
    )
    results['coupling_random'] = simulate_enhanced(
        random_offsets, include_wake=False, include_coupling=True,
        include_added_mass=False, include_trim=False, label="Coupling Random"
    )
    
    # Effect 3: Added mass only
    print("4. Added mass only...")
    results['added_mass_sync'] = simulate_enhanced(
        sync_offsets, include_wake=False, include_coupling=False,
        include_added_mass=True, include_trim=False, label="Added Mass Sync"
    )
    results['added_mass_random'] = simulate_enhanced(
        random_offsets, include_wake=False, include_coupling=False,
        include_added_mass=True, include_trim=False, label="Added Mass Random"
    )
    
    # Effect 4: Trim only
    print("5. Trim effects only...")
    results['trim_sync'] = simulate_enhanced(
        sync_offsets, include_wake=False, include_coupling=False,
        include_added_mass=False, include_trim=True, label="Trim Sync"
    )
    results['trim_random'] = simulate_enhanced(
        random_offsets, include_wake=False, include_coupling=False,
        include_added_mass=False, include_trim=True, label="Trim Random"
    )
    
    # All effects combined
    print("6. All effects combined...")
    results['all_sync'] = simulate_enhanced(
        sync_offsets, include_wake=True, include_coupling=True,
        include_added_mass=True, include_trim=True, label="All Effects Sync"
    )
    results['all_random'] = simulate_enhanced(
        random_offsets, include_wake=True, include_coupling=True,
        include_added_mass=True, include_trim=True, label="All Effects Random"
    )
    
    return results


def analyze_effects(results: dict) -> str:
    """Generate analysis report for effect isolation experiment."""
    
    report = []
    report.append("\n" + "=" * 70)
    report.append("SECOND-ORDER EFFECTS ANALYSIS")
    report.append("=" * 70)
    
    baseline_sync = results['baseline_sync'].mean_velocity
    baseline_random = results['baseline_random'].mean_velocity
    
    report.append(f"\nBaseline velocities (no second-order effects):")
    report.append(f"  Synchronized: {baseline_sync:.4f} m/s")
    report.append(f"  Random ±25%:  {baseline_random:.4f} m/s")
    report.append(f"  Difference:   {(baseline_random - baseline_sync)/baseline_sync*100:+.4f}%")
    
    effects = [
        ('Wake Interactions', 'wake'),
        ('Blade Coupling', 'coupling'),
        ('Added Mass', 'added_mass'),
        ('Hull Trim', 'trim'),
        ('ALL COMBINED', 'all')
    ]
    
    report.append("\n" + "-" * 70)
    report.append("EFFECT-BY-EFFECT BREAKDOWN")
    report.append("-" * 70)
    
    for name, key in effects:
        sync_v = results[f'{key}_sync'].mean_velocity
        random_v = results[f'{key}_random'].mean_velocity
        
        sync_change = (sync_v - baseline_sync) / baseline_sync * 100
        random_change = (random_v - baseline_random) / baseline_random * 100
        
        sync_vs_random = (random_v - sync_v) / sync_v * 100
        
        report.append(f"\n{name}:")
        report.append(f"  Synchronized: {sync_v:.4f} m/s (Δ from baseline: {sync_change:+.3f}%)")
        report.append(f"  Random ±25%:  {random_v:.4f} m/s (Δ from baseline: {random_change:+.3f}%)")
        report.append(f"  Random vs Sync: {sync_vs_random:+.4f}%")
        
        # Effect-specific metrics
        if key != 'all':
            sync_result = results[f'{key}_sync']
            random_result = results[f'{key}_random']
            
            if key == 'wake':
                report.append(f"  Avg wake penalty - Sync: {sync_result.wake_penalty_avg*100:.2f}%, Random: {random_result.wake_penalty_avg*100:.2f}%")
            elif key == 'coupling':
                report.append(f"  Avg coupling penalty - Sync: {sync_result.coupling_penalty_avg*100:.2f}%, Random: {random_result.coupling_penalty_avg*100:.2f}%")
            elif key == 'added_mass':
                report.append(f"  Avg added mass effect - Sync: {sync_result.added_mass_effect_avg*100:.2f}%, Random: {random_result.added_mass_effect_avg*100:.2f}%")
            elif key == 'trim':
                report.append(f"  Avg trim drag increase - Sync: {sync_result.trim_drag_increase_avg*100:.3f}%, Random: {random_result.trim_drag_increase_avg*100:.3f}%")
    
    # Summary
    all_sync = results['all_sync'].mean_velocity
    all_random = results['all_random'].mean_velocity
    total_diff = (all_random - all_sync) / all_sync * 100
    
    report.append("\n" + "=" * 70)
    report.append("SUMMARY")
    report.append("=" * 70)
    report.append(f"\nWith ALL second-order effects included:")
    report.append(f"  Synchronized crew: {all_sync:.4f} m/s")
    report.append(f"  Random ±25% crew:  {all_random:.4f} m/s")
    report.append(f"  Difference: {total_diff:+.4f}%")
        
    return "\n".join(report)


def plot_effect_comparison(results: dict, save_path: str = None):
    """Create visualization comparing all effects."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Effect names and keys
    effects = [
        ('Baseline', 'baseline'),
        ('+ Wake', 'wake'),
        ('+ Coupling', 'coupling'),
        ('+ Added Mass', 'added_mass'),
        ('+ Trim', 'trim'),
        ('ALL Effects', 'all')
    ]
    
    # Colors
    sync_color = 'steelblue'
    random_color = 'coral'
    
    for ax, (name, key) in zip(axes.flat, effects):
        sync_result = results[f'{key}_sync']
        random_result = results[f'{key}_random']
        
        # Plot velocity over time (steady state region)
        t_mask = sync_result.time > 15
        
        ax.plot(sync_result.time[t_mask], sync_result.velocity[t_mask], 
                color=sync_color, label=f'Sync: {sync_result.mean_velocity:.4f} m/s', alpha=0.8)
        ax.plot(random_result.time[t_mask], random_result.velocity[t_mask], 
                color=random_color, label=f'Random: {random_result.mean_velocity:.4f} m/s', alpha=0.8)
        
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Effect of Synchronization with Second-Order Hydrodynamic Effects', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_effect_breakdown(results: dict, save_path: str = None):
    """Bar chart showing contribution of each effect."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    baseline_sync = results['baseline_sync'].mean_velocity
    baseline_random = results['baseline_random'].mean_velocity
    
    effects = ['Wake', 'Coupling', 'Added Mass', 'Trim', 'ALL']
    keys = ['wake', 'coupling', 'added_mass', 'trim', 'all']
    
    # Left plot: Absolute velocities
    ax1 = axes[0]
    x = np.arange(len(effects))
    width = 0.35
    
    sync_velocities = [results[f'{k}_sync'].mean_velocity for k in keys]
    random_velocities = [results[f'{k}_random'].mean_velocity for k in keys]
    
    bars1 = ax1.bar(x - width/2, sync_velocities, width, label='Synchronized', color='steelblue')
    bars2 = ax1.bar(x + width/2, random_velocities, width, label='Random ±25%', color='coral')
    
    ax1.axhline(y=baseline_sync, color='steelblue', linestyle='--', alpha=0.5, label=f'Baseline sync: {baseline_sync:.4f}')
    ax1.axhline(y=baseline_random, color='coral', linestyle='--', alpha=0.5, label=f'Baseline random: {baseline_random:.4f}')
    
    ax1.set_xlabel('Effect Included')
    ax1.set_ylabel('Mean Velocity (m/s)')
    ax1.set_title('Absolute Velocity by Effect')
    ax1.set_xticks(x)
    ax1.set_xticklabels(effects)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right plot: Percentage difference (random vs sync)
    ax2 = axes[1]
    
    pct_diffs = [(r - s) / s * 100 for s, r in zip(sync_velocities, random_velocities)]
    colors = ['green' if d > 0 else 'red' for d in pct_diffs]
    
    bars = ax2.bar(effects, pct_diffs, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, pct_diffs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:+.3f}%', ha='center', va='bottom', fontsize=10)
    
    ax2.set_xlabel('Effect Included')
    ax2.set_ylabel('Random vs Sync Difference (%)')
    ax2.set_title('Does Desynchronization Help or Hurt?')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_pitch_comparison(results: dict, save_path: str = None):
    """Compare pitch dynamics between sync and random crews."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    sync = results['trim_sync']
    random = results['trim_random']
    
    # Steady state region
    t_mask = sync.time > 20
    
    # Pitch angle over time
    ax1 = axes[0, 0]
    ax1.plot(sync.time[t_mask], np.degrees(sync.pitch_angle[t_mask]), 
             'b-', label='Synchronized', alpha=0.8)
    ax1.plot(random.time[t_mask], np.degrees(random.pitch_angle[t_mask]), 
             'r-', label='Random ±25%', alpha=0.8)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Pitch Angle (degrees)')
    ax1.set_title('Hull Pitch Angle')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Pitch histogram
    ax2 = axes[0, 1]
    ax2.hist(np.degrees(sync.pitch_angle[t_mask]), bins=50, alpha=0.6, 
             label=f'Sync (std={np.std(np.degrees(sync.pitch_angle[t_mask])):.3f}°)', color='blue')
    ax2.hist(np.degrees(random.pitch_angle[t_mask]), bins=50, alpha=0.6,
             label=f'Random (std={np.std(np.degrees(random.pitch_angle[t_mask])):.3f}°)', color='red')
    ax2.set_xlabel('Pitch Angle (degrees)')
    ax2.set_ylabel('Count')
    ax2.set_title('Pitch Angle Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Velocity comparison
    ax3 = axes[1, 0]
    ax3.plot(sync.time[t_mask], sync.velocity[t_mask], 'b-', label='Synchronized', alpha=0.8)
    ax3.plot(random.time[t_mask], random.velocity[t_mask], 'r-', label='Random ±25%', alpha=0.8)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity with Trim Effects')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Phase space (pitch angle vs angular velocity)
    ax4 = axes[1, 1]
    # Calculate angular velocity from pitch angle
    dt = sync.time[1] - sync.time[0]
    sync_omega = np.gradient(sync.pitch_angle, dt)
    random_omega = np.gradient(random.pitch_angle, dt)
    
    ax4.plot(np.degrees(sync.pitch_angle[t_mask]), np.degrees(sync_omega[t_mask]), 
             'b.', alpha=0.1, markersize=1, label='Synchronized')
    ax4.plot(np.degrees(random.pitch_angle[t_mask]), np.degrees(random_omega[t_mask]), 
             'r.', alpha=0.1, markersize=1, label='Random')
    ax4.set_xlabel('Pitch Angle (degrees)')
    ax4.set_ylabel('Angular Velocity (deg/s)')
    ax4.set_title('Pitch Phase Space')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def run_monte_carlo_enhanced(n_trials: int = 30):
    """Monte Carlo with all effects enabled."""
    
    print(f"\nRunning Monte Carlo with all effects ({n_trials} trials)...")
    
    sync_velocities = []
    random_velocities = []
    
    for i in range(n_trials):
        # Random offsets with varying spread
        max_offset = np.random.uniform(0.1, 0.35)
        random_offsets = list(np.random.uniform(-max_offset, max_offset, 20))
        
        # Synchronized
        sync_result = simulate_enhanced(
            [0.0] * 20,
            include_wake=True, include_coupling=True,
            include_added_mass=True, include_trim=True,
            duration=20.0
        )
        sync_velocities.append(sync_result.mean_velocity)
        
        # Random
        random_result = simulate_enhanced(
            random_offsets,
            include_wake=True, include_coupling=True,
            include_added_mass=True, include_trim=True,
            duration=20.0
        )
        random_velocities.append(random_result.mean_velocity)
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_trials} trials...")
    
    sync_velocities = np.array(sync_velocities)
    random_velocities = np.array(random_velocities)
    
    print(f"\nMonte Carlo Results (with all second-order effects):")
    print(f"  Synchronized - Mean: {np.mean(sync_velocities):.4f}, Std: {np.std(sync_velocities):.4f}")
    print(f"  Random - Mean: {np.mean(random_velocities):.4f}, Std: {np.std(random_velocities):.4f}")
    
    # Paired comparison
    differences = random_velocities - sync_velocities
    pct_differences = differences / sync_velocities * 100
    
    print(f"\n  Paired differences:")
    print(f"    Mean: {np.mean(pct_differences):+.4f}%")
    print(f"    Std: {np.std(pct_differences):.4f}%")
    print(f"    Range: [{np.min(pct_differences):+.4f}%, {np.max(pct_differences):+.4f}%]")
    
    return sync_velocities, random_velocities, pct_differences


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":    
    # Run effect isolation experiment
    results = run_effect_isolation_experiment()
    
    # Generate analysis
    analysis = analyze_effects(results)
    print(analysis)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_effect_comparison(results, "figures/effect_comparison.png")
    plot_effect_breakdown(results, "figures/effect_breakdown.png")
    plot_pitch_comparison(results, "figures/pitch_comparison.png")
    
    # Monte Carlo
    sync_v, random_v, pct_diff = run_monte_carlo_enhanced(n_trials=30)
    
    # Plot Monte Carlo results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pct_diff, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(x=np.mean(pct_diff), color='red', linestyle='--', 
               label=f'Mean: {np.mean(pct_diff):+.3f}%')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax.set_xlabel('Random vs Sync Velocity Difference (%)')
    ax.set_ylabel('Count')
    ax.set_title('Monte Carlo: Effect of Desynchronization\n(All Second-Order Effects Included)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig("figures/monte_carlo_enhanced.png", dpi=150, bbox_inches='tight')
    
    # Save full report
    with open("figures/second_order_effects_report.txt", "w") as f:
        f.write("DRAGON BOAT: SECOND-ORDER HYDRODYNAMIC EFFECTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(analysis)
        f.write(f"\n\nMONTE CARLO RESULTS ({len(pct_diff)} trials):\n")
        f.write(f"Mean difference (random vs sync): {np.mean(pct_diff):+.4f}%\n")
        f.write(f"Std deviation: {np.std(pct_diff):.4f}%\n")
        f.write(f"Range: [{np.min(pct_diff):+.4f}%, {np.max(pct_diff):+.4f}%]\n")