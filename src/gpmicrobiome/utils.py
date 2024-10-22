"""utils.py."""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pandas as pd
from numpyro.diagnostics import summary
from numpyro.infer import MCMC


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
                            *jnp.unravel_index(jnp.arange(values.size), values.shape),
                            strict=True,
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
