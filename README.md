# RC For GFD Emulation

## Last minute writing stuff

Other logistics
- minimize notebook sizes ... things that will go to overleaf etc...
- move auxiliary stuff to trash repo


## After sending off the draft

- textsize in figures
- update RC fig to have T anom?
- re-optimize the Gulf case
- Play with the linear case...


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
