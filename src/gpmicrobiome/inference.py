"""inference.py."""
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import Array, random
from numpyro.infer import MCMC, NUTS

from gpmicrobiome.constants import GP_INPUT_DIMENSIONS, OTU_MATRIX_DIMENSIONS
from gpmicrobiome.models import get_default_priors, model

# ruff: noqa: N803, PLR0913, PLR0917

KeyArray = Array


def run_nuts(
    key: KeyArray,
    X: Array,
    y: Array,
    X_pred: Array | None = None,
    reference_category: int | Array | None = None,
    priors: dict[str, dist.Distribution] | None = None,
    num_basis: int = 10,
    num_warmup: int = 1_000,
    num_samples: int = 1_000,
    num_chains: int = 4,
    use_deterministic: bool = False,
    consider_zero_inflation: bool = True,
) -> MCMC:
    """Run GPMicrobiome.

    HMC with NUTS.

    Args:
        key: PRNGKey.
        X: Measurement time points (n_timepoints by 1).
        y: OTU counts (n_timepoints by n_otus).
        X_pred: Prediction time points (n_prediction_timepoints by 1).
        reference_category: Reference category to make the softmax transformation bijective.
            If not provided, then we take the middle OTU with respect to counts, i.e.
            `reference_category = jnp.argsort(jnp.sum(y, 0))[y.shape[1] // 2]`.
        priors: Priors distributions. Defaults to gpmicrobiome.src.get_default_priors().
        num_basis: Number of basis functions. Defaults to 10.
        num_warmup: Number of warmup iterations. Defaults to 1_000.
        num_samples: Number of sampling iterations. Defaults to 1_000.
        num_chains: Number of chains. Defaults to 4.
        use_deterministic: Whether to include the deterministic sites in trace. Defaults to False.
        consider_zero_inflation: Whether to consider zero inflation. Defaults to True.

    Returns:
        NumPyro MCMC object.
    """
    assert len(y.shape) == OTU_MATRIX_DIMENSIONS, "y should have two dimensions"
    assert len(X.shape) == GP_INPUT_DIMENSIONS, "X should have two dimensions"
    if X_pred is not None:
        assert (
            len(X_pred.shape) == GP_INPUT_DIMENSIONS
        ), "X_pred should have two dimensions when provided"

    reference_category = reference_category or (
        jnp.argsort(jnp.sum(y, 0))[y.shape[1] // 2]
    ).astype(int)

    priors = get_default_priors() | (priors if priors is not None else {})

    key, key_ = random.split(key, 2)
    nuts_kernel = NUTS(model)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )
    key, key_ = random.split(key, 2)
    mcmc.run(
        key_,
        X=X,
        y=y,
        num_basis=num_basis,
        priors=priors,
        reference_category=reference_category,
        X_pred=X_pred,
        use_deterministic=use_deterministic,
        consider_zero_inflation=consider_zero_inflation,
    )
    return mcmc
