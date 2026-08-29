# Result-to-output crosswalk

The following checks define a successful reproduction.

## Public model panel

- 56 models from five developers in the source panel;
- 32 dense models from 11 multi-model generations in the fixed-effects fit;
- common slope `0.2263410455`;
- generation-clustered standard error `0.1295417024`;
- all 11 generation slopes below one;
- the smallest sibling more token-intensive in every generation;
- contrasts ranging from `1.75` to `48.6779` (reported as `1.8` to `48.7`).

## Llama 3 calibration

- constrained/unconstrained RMSE approximately `.00397`/`.00390`;
- flagship optima approximately `393B, 16.1T` and `622B, 10.2T`;
- 8B total wedges `114.8T` and `150.9T`;
- 70B total wedges `33.0T` and `55.0T`;
- common data wedges `-4.34B` and `-4.16B`;
- the ordering and negative common data wedge survive the reported local-rate
  sensitivity grid;
- constrained leave-one-budget-out ranges reproduce the appendix values.

`python verify.py` implements these checks with explicit numerical tolerances.
