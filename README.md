# Dragon Boat Synchronization: A Computational Analysis

**Does paddle synchronization actually affect dragon boat velocity?**

This repository contains physics-based simulations investigating the hydrodynamic effects of paddle timing in dragon boat racing. 

## Key Findings

### TL;DR
- **Simplified model**: Synchronization has negligible effect (< 0.1% velocity difference)
- **With hydrodynamic effects**: Desynchronization provides ~3.5% velocity advantage
- **Dominant mechanism**: Blade-to-blade coupling — synchronized strokes cause all paddles to compete for the same water

### Results Summary

| Effect | Sync vs Random | Winner |
|--------|---------------|--------|
| Wake interactions | +0.37% | Sync |
| Blade-to-blade coupling | +2.67% | **Random** |
| Added mass | +0.42% | Random |
| Hull trim | < 0.01% | Negligible |
| **Combined** | **+3.45%** | **Random** |

## Quick Start

### Requirements

```bash
pip install numpy matplotlib scipy
```

### Run Basic Simulation

```bash
cd src
python dragon_boat_sim.py
```

This runs the simplified model showing synchronization has negligible effect under basic assumptions.

### Run Enhanced Simulation

```bash
python dragon_boat_enhanced.py
```

This includes all second-order hydrodynamic effects:
1. Wake interactions between paddles
2. Blade-to-blade hydrodynamic coupling
3. Added mass during acceleration
4. Hull trim from crew weight shifts

## Physical Model

### Basic Model
- 20 paddlers, each producing half-sine force pulses
- Quadratic hull drag: F = ½ρCdAv²
- Impulse conservation argument shows sync doesn't matter

### Second-Order Effects

**Wake Interactions**: Each stroke leaves a decaying wake that reduces effectiveness of subsequent strokes in the same region.

**Blade Coupling**: When multiple blades are in water simultaneously, they compete for the same incompressible fluid, reducing individual efficiency.

**Added Mass**: Accelerating through water requires also accelerating surrounding fluid (virtual mass).

**Hull Trim**: Crew weight shifts during strokes cause pitching, which affects drag coefficient.

## Parameters

| Parameter | Value |
|-----------|-------|
| Boat mass | 1200 kg |
| Boat length | 12.5 m |
| Wetted area | 15 m² |
| Drag coefficient | 0.01 |
| Paddlers | 20 |
| Peak force/paddler | 250 N |
| Stroke duration | 0.5 s |
| Stroke rate | 60 spm |

## Contributing

Contributions welcome! Particularly interested in:
- CFD validation of coupling parameters
- Experimental data from instrumented paddles
- Extension to other paddle sports (outrigger, kayak)
