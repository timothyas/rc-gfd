# RC For GFD Emulation

## Quick TODO

- "Appendix" figures:
    - [x] For adjacency and input normalization, figure from new experiment
      running
    - [ ] For data normalization, could use `svd` runs with the different
      normalization strategies
    - [ ] Decide on hand waviness of tavgmisfit... change equations in training
      subsection
    - [ ] Figure out why larger than 500 reservoir size is not working
    - [ ] Re-run 12D L96 case with no training spinup...

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

For this it is useful to note the identities related to products of normally
distributed variables
`var(x y) = var(x) var(y) + var(x)E(y)**2 + var(y)E(x)**2`
and
`var(x + y) = var(x) + var(y) + 2 cov(x ... y ... something)`
basically assume that they are uncorrelated, and normalization means that E( )
is zero (expectation of the random matrix is zero by definition).
Then,
`var(w_i^Tu) = sum_j^{n_input} var(w_{i,j}) var(u_j) = n_input`
if we assume that `u~N(0,1)` ... that it's been normalized (and this is by
construction the case for wij).

Note also that it's easier to say something about the variance rather than the
inner product of each row of Win and the input vector because we don't have a
lower bound on `(w_i, u)` ... we just have `(w_i, u)^2 <= ||w_i||^2 ||u||^2||`
but it doesn't really help to say that we're less than something, we want to
show that the lower bound is too high!
