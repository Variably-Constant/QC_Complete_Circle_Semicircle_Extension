# The Complete Circle: Extending $C_{qc}$ to the Conjugate Sector

## Overview

This research extends the previously established semicircle constraint to the **complete circle**, incorporating both the primary sector ($C_{qc} > 0$) and the conjugate sector ($C_{qc} < 0$). This extension unifies matter and antimatter within a single geometric framework and provides testable predictions for quantum experiments.

## Key Results

### Mathematical Theorems (Proven)

1. **Arc Length = π (Semicircle)**: The Fisher information metric gives arc length = π for q ∈ [0, 1]
2. **Complete Circle = 2π**: The full circle parameterization has information-theoretic circumference = 2π
3. **Both Roots Exist**: $C_{qc}^{(\pm)} = \pm\sqrt{q(1-q)}$ are both valid solutions to the circle constraint
4. **CPT Correspondence**: Primary ↔ conjugate mapping corresponds to the CPT transformation

### Experimental Predictions (Testable)

1. **Interference Visibility**: $\mathcal{V}(q) = 2q(1-q)$ — maximum at q = 0.5
2. **CPT Violation Bound**: $\delta_{CPT} < 10^{-18}$
3. **Neutral Meson Oscillation**: Universal relation from complete circle geometry
4. **Neutron-Antineutron Oscillation**: $\tau_{n\bar{n}} \sim 10^8$ s

## File Structure

```
QC-Research-Complete-Circle/
├── README.md                          # This file
├── .gitignore                         # LaTeX artifacts excluded
├── complete_circle_extension.tex      # Main paper (LaTeX)
├── complete_circle_extension.pdf      # Compiled paper
└── tests/
    ├── test_visibility_vs_q.py        # IonQ visibility experiment
    ├── test_visibility_vs_q.json      # Results
    ├── test_arc_length_verification.py # Mathematical verification
    └── test_arc_length_verification.json
```

## Building the Paper

```bash
pdflatex complete_circle_extension.tex
pdflatex complete_circle_extension.tex  # Run twice for references
```

## Running Experiments

### Local Simulation (No Hardware Required)

```bash
cd tests
python test_visibility_vs_q.py --local --shots 1000
python test_arc_length_verification.py --local
```

### IonQ Hardware via Azure Quantum

```bash
cd tests
python test_visibility_vs_q.py --shots 1000
```

## Connection to Semicircle Constraint

This work extends [QC-Research-Semicircle-Constraint](../QC-Research-Semicircle-Constraint/):

| Property | Semicircle (Previous) | Complete Circle (This Work) |
|----------|----------------------|----------------------------|
| Correlation | $C_{qc} = +\sqrt{q(1-q)}$ | $C_{qc}^{(\pm)} = \pm\sqrt{q(1-q)}$ |
| Arc length | π | 2π |
| Topology | Arc | Circle $S^1$ |
| Physical content | Matter, forward time | Matter + antimatter, CPT unified |

## Theoretical Framework

The complete circle extends the 4DLT (Four-Dimensional Lattice Theory) framework:

- **Primary Sector** ($C_{qc} > 0$): Observable universe, matter, forward time
- **Conjugate Sector** ($C_{qc} < 0$): CPT-conjugate, antimatter correspondence

The angular parameterization:
$$q(\phi) = \frac{1 + \cos\phi}{2}, \quad C_{qc}(\phi) = \frac{1}{2}\sin\phi$$

covers the complete circle for $\phi \in [0, 2\pi)$.

## References

1. Newton, M. "The Semicircle Constraint: Geometric Foundations for Variational Quantum Algorithm Optimization" (2026)
2. Dirac, P.A.M. "The Quantum Theory of the Electron" Proc. R. Soc. Lond. A (1928)
3. Lüders, G. "On the Equivalence of Invariance under Time Reversal and under Particle-Antiparticle Conjugation" (1954)

## Author

Mark Newton
mark@variablyconstant.com

## License

MIT License
