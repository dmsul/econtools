from typing import Union, Optional, Tuple

import pandas as pd
import numpy as np


def binscatter(
    x: Union[str, np.ndarray], y: Union[str, np.ndarray],
    n: int=20, data: Optional[pd.DataFrame]=None,
    discrete: bool=False, median: bool=False
) -> Tuple[np.ndarray, np.ndarray]:
    """Binscatter.

    Args:
        x (array or str): x-axis values. If type ``str``, column in ``data``.
        y (array or str): y-axis values, same length as ``x``. If type ``str``,
            column in ``data``.

    Keyword Args:
        n (int): Default 20. Number of bins.
        discrete (bool): Default False. If True, every unique value in ``x`` is
            given its own bin.
        median (bool): Default False. Calculate the median for each bin instead
            of the mean. Only applies to y-axis values.

    Returns:
        tuple:
            * **x_bin_value** (*array*) - Array of x bin values.
            * **y_bin_value** (*array*) - Array of y bin values.
    """

    # If no `data` is passed, assume arrays
    if (isinstance(data, pd.DataFrame) and
            isinstance(x, str) and
            isinstance(y, str)):
        x = data[x]
        y = data[y]

    if discrete:
        x_bin_id = x
        x_bin_value = np.unique(x_bin_id)
    else:
        x_bin_id = pd.qcut(x, n)
        x_bin_value = pd.DataFrame(x).groupby(x_bin_id).mean()

    if median:
        y_bin_value = pd.DataFrame(y).groupby(x_bin_id).median()
    else:
        y_bin_value = pd.DataFrame(y).groupby(x_bin_id).mean()

    return x_bin_value, y_bin_value


def legend_below(ax, *args, **kwargs) -> None:
    """Create a legend below and outside the main axis object.

    Args:
        ax (Axis): The main ``Axis`` object.
        *args: other args to pass to ``ax.legend``
        **kwargs: other keyword args to pass to ``ax.legend``

    Keyword Args:
        shrink (bool): Default False. Should be True.
        anchor (tuple): 2-tuple to pass to `bbox_to_anchor`. This
            aligns the legend to the rest of the Axis. If you need more space
            between the legend and your figure, make the second digit more
            negative.

    Returns:
        None:
    """
    shrink = kwargs.pop('shrink', False)
    anchor = kwargs.pop('anchor', (0.5, -0.1))

    if shrink:
        shrink_axes_for_legend(ax)
    # Put legend centered, just below axes
    ax.legend(*args, loc='upper center', bbox_to_anchor=anchor, **kwargs)


def shrink_axes_for_legend(*args) -> None:
    for ax in args:
        box = ax.get_position()
        new_box = [box.x0, box.y0 + box.height * 0.1,
                   box.width, box.height * 0.9]
        ax.set_position(new_box)
