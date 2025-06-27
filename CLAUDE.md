# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python implementation of the Tolman-Oppenheimer-Volkoff (TOV) equations for neutron stars in Scalar-Tensor Theories (STT) of gravity. The codebase solves for nonrotating relativistic stars with piecewise polytropic equations of state in modified gravity theories.

## Code Architecture

### Core Components

- **tov_stt_solver.py**: Complete standalone Python module containing the `TOVSTTSolver` class
- **TOV-pp-STT.ipynb**: Jupyter notebook with original implementation and detailed physics explanations
- Both implementations solve the same physics but with different organization

### Key Classes and Functions

- `TOVSTTSolver`: Main solver class encapsulating all functionality
  - Handles equation of state setup (piecewise polytropic with SLy EOS)
  - Implements TOV differential equations for interior and exterior regions
  - Performs boundary condition optimization using scipy.optimize
  - Generates complete stellar solutions with physical property calculations

### Physics Implementation

- **Equation of State**: 7-segment piecewise polytropic model with SLy high-density EOS
- **TOV System**: 5 coupled ODEs (pressure, mass, metric functions, scalar field)
- **Scalar-Tensor Theory**: Coupling parameter β controls deviation from General Relativity
- **Boundary Conditions**: Optimized to satisfy asymptotic conditions (ν→0, φ→0 at infinity)

## Development Commands

### Running the Code

```bash
# Execute the standalone script
python tov_stt_solver.py

# Run the Jupyter notebook
jupyter notebook TOV-pp-STT.ipynb
```

### Dependencies

The code requires standard scientific Python libraries:
- numpy, scipy, matplotlib
- For notebooks: jupyter, IPython

### Testing and Validation

- Solutions are validated by checking asymptotic behavior
- Physical properties (mass, radius, compactness) are computed and verified
- Numerical convergence is monitored through grid resolution studies
- Expected typical results: M ≈ 1.7 M☉, R ≈ 11.6 km for central density ~10^15 g/cm³

## File Structure

- Geometrized units (G=c=M_sun=1) used internally
- CGS unit outputs provided for physical interpretation
- Solutions saved as .dat files with both unit systems
- Comprehensive plotting capabilities for all solution components

## Key Parameters

- **Central density**: Typically 10^14 - 10^16 g/cm³
- **β parameter**: Scalar-tensor coupling (-4.5 typical for viable theories)
- **Grid resolution**: Default 32001 points for high accuracy
- **Integration**: LSODA adaptive solver with tight tolerances (rtol=1e-12)