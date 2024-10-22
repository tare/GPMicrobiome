"""test_gpmicrobiome.py."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import random
from numpyro.infer import MCMC, NUTS

from gpmicrobiome import run_nuts
from gpmicrobiome.utils import get_mcmc_summary

SAMPLING_ZERO_B = 1e-2
NON_SAMPLING_ZERO_B = 0.5


def test_gpmicrobiome_dims() -> None:
    """Check dimensions without X_pred."""
    num_timepoints = 5
    num_otus = 3
    x = jnp.arange(num_timepoints)[:, None]
    y = jnp.ones((num_timepoints, num_otus))

    key = random.PRNGKey(0)
    mcmc = run_nuts(
        key, x, y, num_warmup=10, num_samples=10, num_chains=1, use_deterministic=True
    )

    posterior_samples = mcmc.get_samples()

    assert posterior_samples["theta_b"].shape == (10, *y.shape)
    assert posterior_samples["theta"].shape == (10, *y.shape)
    assert "theta_test" not in posterior_samples


def test_gpmicrobiome_w_zero_inflation() -> None:
    """Check that we can enable the zero inflation component."""
    num_timepoints = 5
    num_otus = 3
    x = jnp.arange(num_timepoints)[:, None]
    y = jnp.ones((num_timepoints, num_otus))

    key = random.PRNGKey(0)
    mcmc = run_nuts(
        key,
        x,
        y,
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        use_deterministic=True,
        consider_zero_inflation=True,
    )

    posterior_samples = mcmc.get_samples()

    assert "b" in posterior_samples
    assert "theta_b" in posterior_samples


@pytest.mark.parametrize("prediction", [True, False])
def test_gpmicrobiome_w_deterministic(prediction: bool) -> None:  # noqa: FBT001
    """Check that we can store deterministic sites."""
    num_timepoints = 5
    num_pred_timepoints = 5
    num_otus = 3
    x = jnp.arange(num_timepoints)[:, None]
    y = jnp.ones((num_timepoints, num_otus))
    if prediction:
        num_pred_timepoints = 5
        x_pred = jnp.arange(num_timepoints, num_timepoints + num_pred_timepoints)[
            :, None
        ]
    else:
        x_pred = None

    key = random.PRNGKey(0)
    mcmc = run_nuts(
        key,
        x,
        y,
        x_pred,
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        use_deterministic=True,
        consider_zero_inflation=True,
    )

    posterior_samples = mcmc.get_samples()

    assert "g" in posterior_samples
    assert "g_n" in posterior_samples
    assert "g_appended" in posterior_samples
    assert "g_n_appended" in posterior_samples
    assert "theta_n" in posterior_samples
    assert "theta_b" in posterior_samples


@pytest.mark.parametrize("prediction", [True, False])
def test_gpmicrobiome_wout_deterministic(prediction: bool) -> None:  # noqa: FBT001
    """Check that we can discard deterministic sites."""
    num_timepoints = 5
    num_otus = 3
    x = jnp.arange(num_timepoints)[:, None]
    y = jnp.ones((num_timepoints, num_otus))
    if prediction:
        num_pred_timepoints = 5
        x_pred = jnp.arange(num_timepoints, num_timepoints + num_pred_timepoints)[
            :, None
        ]
    else:
        x_pred = None

    key = random.PRNGKey(0)
    mcmc = run_nuts(
        key,
        x,
        y,
        x_pred,
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        use_deterministic=False,
        consider_zero_inflation=True,
    )

    posterior_samples = mcmc.get_samples()

    assert "g" not in posterior_samples
    assert "g_n" not in posterior_samples
    assert "g_appended" not in posterior_samples
    assert "g_n_appended" not in posterior_samples
    assert "theta_n" not in posterior_samples
    assert "theta_b" not in posterior_samples


def test_gpmicrobiome_wout_zero_inflation() -> None:
    """Check that we can disable the zero inflation component."""
    num_timepoints = 5
    num_otus = 3
    x = jnp.arange(num_timepoints)[:, None]
    y = jnp.ones((num_timepoints, num_otus))

    key = random.PRNGKey(0)
    mcmc = run_nuts(
        key,
        x,
        y,
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        use_deterministic=True,
        consider_zero_inflation=False,
    )

    posterior_samples = mcmc.get_samples()

    assert "b" not in posterior_samples
    assert "theta_b" not in posterior_samples


def test_gpmicrobiome_dims_pred() -> None:
    """Check dimensions with X_pred."""
    num_timepoints = 5
    num_pred_timepoints = 5
    num_otus = 3
    x = jnp.arange(num_timepoints)[:, None]
    y = jnp.ones((num_timepoints, num_otus))
    x_pred = jnp.arange(num_timepoints, num_timepoints + num_pred_timepoints)[:, None]

    key = random.PRNGKey(0)
    mcmc = run_nuts(
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
    y = y.at[num_timepoints // 2, 0].set(0)  # noqa: PD008

    sampling_zeros = (
        jnp.zeros((num_timepoints, num_otus), dtype=bool)  # noqa: PD008
        .at[num_timepoints // 2, 0]
        .set(True)
    )

    key = random.PRNGKey(0)
    mcmc = run_nuts(
        key, x, y, num_warmup=100, num_samples=100, num_chains=1, use_deterministic=True
    )

    posterior_samples = mcmc.get_samples()

    assert all(posterior_samples["b"].mean(0)[sampling_zeros] < SAMPLING_ZERO_B)
    assert all(posterior_samples["b"].mean(0)[~sampling_zeros] > NON_SAMPLING_ZERO_B)


def test_gp_microbiome_convergence() -> None:
    """Test convergence.

    This is not a universal test for convergence, i.e. this tests get_mcmc_summary() with a toy example.
    """
    num_timepoints = 5
    num_pred_timepoints = 5
    num_otus = 3
    x = jnp.arange(num_timepoints)[:, None]
    y = jnp.ones((num_timepoints, num_otus))
    x_pred = jnp.arange(num_timepoints, num_timepoints + num_pred_timepoints)[:, None]

    key = random.PRNGKey(0)
    mcmc = run_nuts(
        key,
        x,
        y,
        x_pred,
        num_warmup=500,
        num_samples=500,
        num_chains=4,
        use_deterministic=True,
    )

    assert get_mcmc_summary(mcmc).query("r_hat > 1.05").shape[0] == 0


def test_get_mcmc_summary() -> None:
    """Test get_mcmc_summary()."""
    num_obs = 10

    def model() -> None:
        mu = numpyro.sample("mu", dist.Normal(0, 1))
        with numpyro.plate("N", num_obs):
            numpyro.sample("y", dist.Normal(mu, 1))

    key = random.PRNGKey(0)
    key, key_ = random.split(key, 2)
    nuts_kernel = NUTS(model)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=500,
        num_samples=500,
        num_chains=4,
    )
    key, key_ = random.split(key, 2)
    mcmc.run(key_)

    summary = get_mcmc_summary(mcmc)

    assert tuple(summary.columns) == (
        "variable",
        "index",
        "mean",
        "std",
        "median",
        "5.0%",
        "95.0%",
        "n_eff",
        "r_hat",
    )
    assert summary.shape[0] == 1 + num_obs
    assert summary.query("variable == 'mu'").shape[0] == 1
    assert summary.query("variable == 'y'").shape[0] == num_obs
    assert summary.query("r_hat > 1.05").shape[0] == 0
