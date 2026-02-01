# The Complete Circle: Extending the Semicircle Constraint

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18451570.svg)](https://doi.org/10.5281/zenodo.18451570)

## Overview

This research extends the semicircle constraint to the **complete circle**, incorporating both the primary sector (C_qc > 0) and the conjugate sector (C_qc < 0). The mathematical extension is rigorously proven; physical interpretations connecting the conjugate sector to antimatter/CPT are presented as conjectures.

## Key Results

### Proven Theorems

| Theorem | Statement | Status |
|---------|-----------|--------|
| 1 | Arc length of semicircle = pi | **PROVEN** |
| 2 | Complete circle circumference = 2pi | **PROVEN** |
| 3 | Circle parameterization: q(phi) = (1 + cos phi)/2, C_qc(phi) = sin(phi)/2 | **PROVEN** |
| 4 | Both roots C_qc^(+/-) = +/-sqrt(q(1-q)) are valid | **PROVEN** |
| 5 | Visibility formula V(q) = 2q(1-q) | **PROVEN** |

### Conjectures (Priority Claims)

| Conjecture | Statement | Status |
|------------|-----------|--------|
| 1 | CPT correspondence: negative root maps to antimatter | **CONJECTURED** |
| 2 | Missing pi = conjugate sector (matter-antimatter asymmetry) | **CONJECTURED** |
| 3 | Dirac spinor analogy: two roots analogous to electron/positron | **CONJECTURED** |
| 4 | Cosmological selection: universe selects primary sector | **CONJECTURED** |
| 5 | Emergent pi: circular geometry propagates from quantum to cosmic scales | **CONJECTURED** |

## Mathematical Framework

### The Semicircle Constraint (Proven)

From the Born rule and quantum state normalization:

```
(q - 1/2)^2 + C_qc^2 = 1/4
```

where q is measurement probability and C_qc = sqrt(q(1-q)) is the quantum-classical correlation.

### Extension to Complete Circle (Proven)

The constraint admits two solutions:

```
C_qc^(+) = +sqrt(q(1-q))    (Primary sector)
C_qc^(-) = -sqrt(q(1-q))    (Conjugate sector)
```

### Angular Parameterization (Proven)

The complete circle is parameterized by phi in [0, 2pi):

```
q(phi) = (1 + cos(phi)) / 2
C_qc(phi) = sin(phi) / 2
```

- Primary sector: phi in [0, pi], C_qc >= 0
- Conjugate sector: phi in (pi, 2pi), C_qc < 0

### Arc Length Calculation (Proven)

Using the Fisher information metric I_F(q) = 1/(q(1-q)):

```
L_semicircle = integral_0^1 dq / (2*sqrt(q(1-q))) = pi
L_complete = 2pi
```

This uses the beta function identity B(1/2, 1/2) = pi.

### Visibility Formula (Proven)

Interference visibility as a function of q:

```
V(q) = 2q(1-q)
```

Maximum visibility V_max = 0.5 occurs at q = 0.5.

## File Structure

```
QC-Research-Complete-Circle/
├── README.md                              # This file
├── .gitignore                             # LaTeX artifacts excluded
├── complete_circle_extension.tex          # Main paper (LaTeX)
├── complete_circle_extension_nounicode.tex # No-unicode version
├── complete_circle_extension.pdf          # Compiled paper
└── tests/
    ├── test_visibility_vs_q.py            # Visibility experiment
    ├── test_visibility_vs_q.json          # Results
    ├── test_arc_length_verification.py    # Arc length verification
    └── test_arc_length_verification.json  # Results
```

## Building the Paper

```bash
pdflatex complete_circle_extension.tex
pdflatex complete_circle_extension.tex  # Run twice for references
```

For systems without unicode support:
```bash
pdflatex complete_circle_extension_nounicode.tex
```

## Running Verification Tests

### Arc Length Verification (Mathematical)

```bash
cd tests
python test_arc_length_verification.py --local
```

This verifies:
1. Angular substitution method: L = pi
2. Direct integration: integral_0^1 dq/sqrt(q(1-q)) = pi
3. Beta function identity: B(1/2, 1/2) = pi
4. Complete circle circumference = 2pi

### Visibility vs q Test

```bash
cd tests
python test_visibility_vs_q.py --local --shots 1000
```

Tests the prediction V(q) = 2q(1-q) with:
- 9 q values from 0.1 to 0.9
- Maximum visibility at q = 0.5
- Symmetry: V(q) = V(1-q)

## Connection to Semicircle Constraint

This work extends [QC-Research-Semicircle-Constraint](../QC-Research-Semicircle-Constraint/):

| Property | Semicircle (Previous) | Complete Circle (This Work) |
|----------|----------------------|----------------------------|
| Correlation | C_qc = +sqrt(q(1-q)) | C_qc^(+/-) = +/-sqrt(q(1-q)) |
| Arc length | pi | 2pi |
| Topology | Arc | Circle S^1 |
| Physical content | Matter | Matter + conjugate sector |

## Theoretical Summary

### What is Proven

1. **Mathematical structure**: The complete circle with both roots is rigorously established
2. **Information geometry**: Arc length = pi (semicircle), 2pi (complete circle)
3. **Parameterization**: Angular coordinates cover the full circle
4. **Visibility**: V(q) = 2q(1-q) follows from quantum mechanics

### What is Conjectured

1. **Physical interpretation**: The conjugate sector's connection to antimatter/CPT
2. **Cosmological significance**: Why we observe only the primary sector
3. **Dirac analogy**: Whether the two roots mirror electron/positron solutions
4. **Emergent pi geometry**: The value pi emerges from quantum probability space, and this geometric principle propagates across all scales:
   - Quantum: Wave functions oscillate with period 2pi
   - Microscopic: Spherical atomic orbitals and harmonics
   - Macroscopic: Spherical stars, planets, moons
   - Cosmic: Spiral galaxies, elliptical orbits, large-scale structure
5. **π → Fractals connection**: Since circles are scale-invariant, and q = 0.5 is where circular geometry is maximally expressed, fractal patterns naturally emerge there. See [Fractals, Coherence, and Geometric Measurement](../QC-Research-Fractals_Coherence_GeometricMeansurement/)

These conjectures establish priority for future experimental and theoretical investigation.

## Verification Results

### Arc Length Test

```
Method                  | Result     | Expected | Status
------------------------|------------|----------|--------
Angular substitution    | 3.141593   | pi       | PASS
Direct integration      | 3.141593   | pi       | PASS
Beta function B(1/2,1/2)| 3.141593   | pi       | PASS
Complete circle         | 6.283185   | 2pi      | PASS
```

### Visibility Test

```
q     | V(q) Theory | V(q) Measured | Status
------|-------------|---------------|--------
0.10  | 0.1800      | ~0.18         | PASS
0.20  | 0.3200      | ~0.32         | PASS
0.30  | 0.4200      | ~0.42         | PASS
0.40  | 0.4800      | ~0.48         | PASS
0.50  | 0.5000      | ~0.50         | PASS (Maximum)
0.60  | 0.4800      | ~0.48         | PASS
0.70  | 0.4200      | ~0.42         | PASS
0.80  | 0.3200      | ~0.32         | PASS
0.90  | 0.1800      | ~0.18         | PASS
```

Symmetry V(q) = V(1-q) verified.

## Citation

```bibtex
@misc{newton2026completecircle,
  author       = {Newton, Mark},
  title        = {The Complete Circle: Extending the Semicircle Constraint to 2π with Conjectured CPT Correspondence},
  year         = {2026},
  doi          = {10.5281/zenodo.18451570},
  url          = {https://doi.org/10.5281/zenodo.18451570}
}
```

## References

1. Newton, M. "The Semicircle Constraint: A Geometric Framework for Quantum-Classical Correlation" (2026) [DOI: 10.5281/zenodo.18451496](https://doi.org/10.5281/zenodo.18451496)
2. Dirac, P.A.M. "The Quantum Theory of the Electron" Proc. R. Soc. Lond. A (1928)
3. Luders, G. "On the Equivalence of Invariance under Time Reversal and under Particle-Antiparticle Conjugation" (1954)
4. Fisher, R.A. "Theory of Statistical Estimation" Proc. Cambridge Phil. Soc. (1925)

## Author

Mark Newton
mark@variablyconstant.com

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
