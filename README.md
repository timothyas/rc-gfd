# RC For GFD Emulation

## NVAR TODO:

- What does the spatial plot look like, just keep 12th hour predictions and
  truth
- How about KE error, need to represent all timesteps?
- linear results...
- what do KE plots look like with 4, 8, 12 hr timestamps
- Nsub=48


Wishlist:
- Nsub=4, totspec=5e-4 to see if "moderate" penalty in this case is worse than
  the two extremes, or if this is only the case with KE NRMSE

## RC Results

Flow:
- All RC runs are stable
- But overarching question: do we need to run at the model timestep? How well
  can we represent with time skipping? Can spectrum weak constraint be a
  solution and is it necessary (or can we do nmacro=100)?
- As noted in previous literature, and experienced here, hyperparameters
  dramatically change performance
- Here, optimize to obtain consistent results
- Consider Platt etal, optimize KE Density spectrum
- Show with dt=300 (for now) how using spectrum improves results in terms of
  NRMSE and KERMSE (and KERMSE?)
    * Any noticeable trend in the hyperparameters arising that allows us to get
      a better spectral representation?
- Note dependence of solution on regularization (does this change when we use
  totspectral?)
- Compare nsub with best regularization from each
- is nsub=1 necessarily the best?
- Note comparison to NVAR: mostly solutions are overly diffuse... penalizing
  keNrmse will make some sims overly energetic, but more overly smooth... but
  this is by design since we're optimizing so only finding stable configurations


### Results from using KE RMSE (not normalized)

Nsub=16
- Best spectrum is actually with gamma=0
- Can't penalize KE RMSE enough to approach this result
- Intermediate values of spectral penalty produce better NRMSE, but worst
  spectrum...
- Possibly because this can't be resolved, but maybe this is just a bad metric?

Nsub=4
- See tradeoff between NRMSE and spectrum very clearly
- Note that there is little changing in the results with KE penalty... basically
  to get the spectrum better, do less with the dynamics

Comparing Nsub: if we grab the best spectral representation, we don't see a big
difference between each Nsub.
And the Nsub=16 result has better NRMSE... so this actually makes the case for
temporal subsampling.

### Results using KE NRMSE (normalized by standard deviation in time)

Nsub=16
- Considering extremes (gamma = 0 or gamma = 0.1), we get a tradeoff between
  NRMSE and spectrum

Nmacro=100:
- Interestingly this only improves NRMSE slightly, and reduces spectral
  representation slightly... converges to the same thing as using a small
  spectral constraint

Comparing Nsub: we do see an impact from temporal subsampling.

### Comparing the two macro cost functions

- With KE NRMSE, we don't see the same "saturation" where beyond a certain point all results
  are the same (at least we haven't reached that point...)
- In a sense this is troubling because we don't know what this will converge to
- On top of this, getting a better spectral representation really means that we
  do better in the small scales but get an "overly energetic" mid/larger scales
- In another sense, it seems like KE RMSE can't really distinguish between
  spectral errors as well (this claim is pending the Nsub=1 case... since it
  could do this for Nsub=4... but in general this metric is less sensitive)
- It seems like using moderate penalty values performs worse than when we hit
  the extremes.
    * for instance using Nsub=16 and moderate gamma values results in improved
      NRMSE, but worse spectrum ... what's up with that?


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
