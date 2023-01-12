# RC For GFD Emulation

## Reading list


### General background
- Nonnenmacher & Greenberg, (2021). Deep Emulators for Differentiation, Forecasting, and Parametrization in Earth Science Simulators.
    - What architecture(s)? Seems like they learn LEs - is that true?
- Weyn et al (2019, 2020, and 2021).
    - Why don't they use the ConvLSTM? Why CNN? Good to know these...


### Multi scale

- Na et al (2021). Hierarchical delay-memory ESN.
    - Do they really get multiscale performance?
- Moon et al (2021). Hierarchical architectures in RC
    - multiscale?
- Faranda et al (2021). Enhancing geophysical flow machine learning performance
  via scale separation.
    - Multi scale...

### Other RC
- Verstraeten et al, (2007). An experimental unification of RC
    - This will finally help clarify RC terminology (ESN, LSM, RC, etc..) ... I
      hope
    - LE as a validation metric... what did we do that's new again?

## Future

See [this notebook](notebooks/plot_input_contributions.ipynb) and the L96
results.
Basically, I can't make theoretical sense of why normalizing the input matrix by the largest
singular value should work, but the results with L96 look good.
Theoretically, normalizing by `sqrt(n_input)` "makes sense" but performs poorly.
Now I have a theory that the largest singular value works for the lorenz system
because with `n_input=6` this is close to the full range, `max-min`, of input
contributions to the reservoir, and so the input signal is with highly
probability well on the linear regime of the tanh function.
Then, the reservoir and adjacency matrix handles the nonlinearity.
To test this out at least a bit better, would want to use business as usual,
sqrt(nin), and largest singular value normalization for input matrix with the
following two setups:
1. Same system, growing reservoir size
2. Same reservoir size, growing input size
