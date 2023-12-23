"""test_gpmicrobiome.py."""
import jax.numpy as jnp
from gpmicrobiome import run
from jax import random

SAMPLING_ZERO_B = 1e-2
NON_SAMPLING_ZERO_B = 0.5


def test_gpmicrobiome_dims() -> None:
    """Check dimensions without X_pred."""
    num_timepoints = 5
    num_otus = 3
    x = jnp.arange(num_timepoints)[:, None]
    y = jnp.ones((num_timepoints, num_otus))

    key = random.PRNGKey(0)
    mcmc = run(
        key, x, y, num_warmup=10, num_samples=10, num_chains=1, use_deterministic=True
    )

    posterior_samples = mcmc.get_samples()

    assert posterior_samples["theta_b"].shape == (10, *y.shape)
    assert posterior_samples["theta"].shape == (10, *y.shape)
    assert "theta_test" not in posterior_samples


def test_gpmicrobiome_dims_pred() -> None:
    """Check dimensions with X_pred."""
    num_timepoints = 5
    num_pred_timepoints = 5
    num_otus = 3
    x = jnp.arange(num_timepoints)[:, None]
    y = jnp.ones((num_timepoints, num_otus))
    x_pred = jnp.arange(num_timepoints, num_timepoints + num_pred_timepoints)[:, None]

    key = random.PRNGKey(0)
    mcmc = run(
        key,
        x,
        y,
        x_pred,
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        use_deterministic=True,
    )

    posterior_samples = mcmc.get_samples()

    assert posterior_samples["theta_b"].shape == (10, *y.shape)
    assert posterior_samples["theta"].shape == (10, *y.shape)
    assert posterior_samples["theta_pred"].shape == (10, num_pred_timepoints, num_otus)


def test_gpmicrobiome_sampling_zero_detection() -> None:
    """Test sampling zero detection."""
    num_timepoints = 10
    num_otus = 3
    x = jnp.arange(num_timepoints)[:, None]
    y = 1000 * jnp.ones((num_timepoints, num_otus))
    y = y.at[num_timepoints // 2, 0].set(0)

    sampling_zeros = (
        jnp.zeros((num_timepoints, num_otus), dtype=bool)
        .at[num_timepoints // 2, 0]
        .set(True)
    )

    key = random.PRNGKey(0)
    mcmc = run(
        key, x, y, num_warmup=100, num_samples=100, num_chains=1, use_deterministic=True
    )

    posterior_samples = mcmc.get_samples()

    assert all(posterior_samples["b"].mean(0)[sampling_zeros] < SAMPLING_ZERO_B)
    assert all(posterior_samples["b"].mean(0)[~sampling_zeros] > NON_SAMPLING_ZERO_B)
