import numpy as np
from numpy.polynomial.polynomial import Polynomial


"""
We need to remove outliers,
interpolate data to solve for missing points
smooth data to remove false positives for breath calculation
wait 10 seconds from sensor connection to start reading data
"""


def ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    # FIX: Replace np.array(data, copy=False) with np.asarray(data)
    data = np.asarray(data)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    # FIX: Replace np.array(alpha, copy=False) with np.asarray(alpha)
    alpha = np.asarray(alpha).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        # FIX: Replace np.array(offset, copy=False) with np.asarray(offset)
        offset = np.asarray(offset).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out


def calc_poly_vals(time_arr, poly):
    poly = Polynomial([])
    out_arr = []
    for i in time_arr:
        out_arr.append(0)
        for x in len(range(poly.coef)):
            out_arr[-1] += poly.coef[x] * pow(i , x)
    return out_arr