"""src.py."""
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import Array, random
from jax.nn import softmax
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS

# ruff: noqa: N803, N806, F841, PLR0913, PLR0917

KeyArray = Array

GP_INPUT_DIMENSIONS = 2
OTU_MATRIX_DIMENSIONS = 2


def get_default_priors() -> dict[str, dist.Distribution]:
    """Get default priors.

    The default priors are:
        mu_g ~ U(-10, 10),
        sigma_g ~ HalfNormal(0.5),
        b ~ Beta(0.8, 0.6),
        alpha ~ Gamma(2, 2)
        length ~ Gamma(10, 10).

    Returns:
        Prior distributions
    """
    return {
        "mu_g": dist.Uniform(-10, 10),
        "sigma_g": dist.HalfNormal(0.5),
        "b": dist.Beta(0.8, 0.4),
        "alpha": dist.Gamma(2, 2),
        "length": dist.Gamma(10, 10),
    }


def gp(x: Array, L: Array, M: list[int], priors: dict[str, dist.Distribution]) -> Array:
    """Hilbert space approximate Bayesian Gaussian process.

    https://doi.org/10.1007/s11222-022-10167-2

    Args:
        x: Input features.
        L: Interval in which the approximation is valid.
        M: Number of basis functions per dimension.
        priors: Prior distributions.

    Returns:
        Sample from Hilbert space approximate Gaussian process.
    """

    def se_spectral_density(w: Array, alpha: Array, length: Array) -> Array:
        """Get spectral density of the squared exponential covariance function.

        Args:
            w: Input features in the frequency domain.
            alpha: Signal magnitude parameter.
            length: Length-scale parameter.

        Returns:
            Spectral density values of the squared exponential covariance functions.
        """
        # see Eq. 1
        D = w.shape[-1]
        c = alpha * jnp.power(jnp.sqrt(2 * jnp.pi), D) * jnp.prod(length, -1)
        e = jnp.exp(-0.5 * jnp.power(w, 2) @ jnp.power(length, 2))
        return c * e

    def diag_spectral_density(alpha: Array, length: Array, L: Array, S: Array) -> Array:
        """Get diagonal matrix of the spectral density evaluated at the square root of the eigenvalues.

        Args:
            alpha: Signal magnitude parameter.
            length: Length-scale parameter.
            L:  Interval in which the approximation is valid.
            S: Set of possible combinations of univariate eigenfunctions over all dimensions.

        Returns:
            Diagonal matrix of the spectral density evaluated at the square root of the eigenvalues
        """
        # see Eq. 9
        sqrt_eigenvalues = jnp.pi * S / 2 / L
        return se_spectral_density(sqrt_eigenvalues, alpha, length)

    def eigenfunctions(x: Array, L: Array, S: Array) -> Array:
        """Get eigenfunction values.

        Args:
            x: Input features.
            L: Interval in which the approximation is valid.
            S: Set of possible combinations of univariate eigenfunctions over all dimensions.

        Returns:
            Eigenfunction values.
        """
        # see Eq. 10
        sqrt_eigenvalues = jnp.pi * S / 2 / L
        return jnp.prod(
            jnp.power(L, -0.5) * jnp.sin(sqrt_eigenvalues * jnp.expand_dims(x + L, -2)),
            -1,
        )

    assert (
        len(x.shape) == GP_INPUT_DIMENSIONS
    ), "x should have two dimensions (obs x features)"
    S = jnp.transpose(
        jnp.asarray(
            jnp.meshgrid(*[jnp.linspace(1.0, i, num=i) for i in M], indexing="ij")
        ).reshape(len(M), -1)
    )

    priors = priors or {}

    alpha = numpyro.sample("alpha", priors["alpha"])
    with numpyro.plate("d", x.shape[-1]):
        length = numpyro.sample("length", priors["length"])
    phi = eigenfunctions(x, L, S)
    spd = jnp.sqrt(diag_spectral_density(alpha, length, L, S))
    with numpyro.plate("basis", S.shape[0]):
        beta = numpyro.sample("beta", dist.Normal(0, 1))
    return numpyro.deterministic("f", phi @ (spd * beta))


def model(
    X: Array,
    y: Array,
    num_basis: int,
    priors: dict[str, dist.Distribution],
    X_pred: Array | None = None,
    reference_category: int | Array = 0,
    use_deterministic: bool = False,
    consider_zero_inflation: bool = True,
) -> None:
    """GPMicrobiome generative model function.

    X values are centered around zero by substracting 0.5 * max(X).

    Args:
        X: Measurement time points.
        y: Observed counts.
        num_basis: Number of basis functions.
        priors: Prior distributions.
        X_pred: Prediction time points.
        reference_category: Reference category to make the softmax transformation bijective. Defaults to zero.
        use_deterministic: Include additional parameters in trace for debugging. Defaults to zero.
        consider_zero_inflation: Whether to consider zero inflation. Defaults to True.
    """
    if X_pred is not None:
        X = jnp.vstack((X, X_pred))

    X = X - jnp.min(X)
    X = X - 0.5 * jnp.max(X)

    L = jnp.asarray([1.5 * jnp.max(X)])
    M = X.shape[-1] * [num_basis]

    with numpyro.plate("otu-1", y.shape[-1] - 1):
        f = numpyro.handlers.scope(gp, "gp")(X, L, M, priors)
        mu_g = numpyro.sample("mu_g", priors["mu_g"])
        sigma_g = numpyro.sample("sigma_g", priors["sigma_g"])

    g = mu_g + f
    if use_deterministic:
        g = numpyro.deterministic("g", g)

    raw_g_n = numpyro.sample("raw_g_n", dist.Normal(jnp.zeros(g.shape), 1.0))
    g_n = g + sigma_g * raw_g_n
    if use_deterministic:
        g_n = numpyro.deterministic("g_n", g_n)

    # add reference category
    g_appended = jnp.insert(g, reference_category, jnp.zeros(g.shape[-2]), 1)
    g_n_appended = jnp.insert(g_n, reference_category, jnp.zeros(g_n.shape[-2]), 1)
    if use_deterministic:
        g_appended = numpyro.deterministic("g_appended", g_appended)
        g_n_appended = numpyro.deterministic("g_n_appended", g_n_appended)

    if X_pred is not None:
        theta = numpyro.deterministic(
            "theta", softmax(g_appended[: -X_pred.shape[-2]], axis=-1)
        )
        theta_n = softmax(g_n_appended[: -X_pred.shape[-2]], axis=-1)
        if use_deterministic:
            theta_n = numpyro.deterministic("theta_n", theta_n)
        theta_pred = numpyro.deterministic(
            "theta_pred", softmax(g_appended[-X_pred.shape[-2] :], axis=-1)
        )
    else:
        theta = numpyro.deterministic("theta", softmax(g_appended, axis=-1))
        theta_n = softmax(g_n_appended, axis=-1)
        if use_deterministic:
            theta_n = numpyro.deterministic("theta_n", theta_n)

    with numpyro.plate("otu", y.shape[-1]), numpyro.plate("time", y.shape[-2]):
        b = numpyro.sample("b", priors["b"])

    if consider_zero_inflation:
        theta_b = (tmp := theta_n * b) / jnp.sum(tmp, axis=-1, keepdims=True)
        if use_deterministic:
            theta_b = numpyro.deterministic("theta_b", theta_b)
    else:
        theta_b = theta_n

    numpyro.sample(
        "y",
        dist.MultinomialProbs(
            probs=theta_b,
            total_count=jnp.sum(y, -1),
            total_count_max=jnp.max(jnp.sum(y, -1)),
        ),
        obs=y,
    )


def get_mcmc_summary(mcmc: MCMC) -> pd.DataFrame:
    """Get MCMC summary DataFrame.

    Args:
        mcmc: MCMC object.

    Returns:
        DataFrame.
    """

    def process_variable(
        variable: str, data: Mapping[str, npt.NDArray[np.float64] | np.float64]
    ) -> dict[str, Any]:
        res: dict[str, str | list[tuple[int, ...]] | list[np.float64] | list[None]] = {
            "variable": variable
        }
        for statistic, values in data.items():
            if "index" not in res:
                if isinstance(values, np.ndarray):
                    res["index"] = [
                        tuple(map(int, x))
                        for x in zip(
                            *jnp.unravel_index(jnp.arange(values.size), values.shape)
                        )
                    ]
                else:
                    res["index"] = [None]
            if isinstance(values, np.ndarray):
                res[statistic] = values.flatten().tolist()
            else:
                res[statistic] = [values]
        return res

    return pd.concat(
        [
            pd.DataFrame.from_dict(process_variable(variable, data))
            for variable, data in summary(mcmc.get_samples(group_by_chain=True)).items()
        ],
        axis=0,
        ignore_index=True,
    )


def run(
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
