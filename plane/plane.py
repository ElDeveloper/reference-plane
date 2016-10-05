from __future__ import division

__credits__ = "Robert Kern"
__url__ = "https://groups.google.com/forum/#!topic/comp.lang.python/0JiqYeo0qu4"

import numpy as np
import pandas as pd

from scipy.spatial.distance import euclidean

def point_to_plane_distance(abcd, point):
    """
    Calculates the euclidean distance from a point to a plane

    Parameters
    ----------
    abcd : array-like
        The four coefficients of an equation that defines a
        plane of the form a*x + b*y + c*z + d = 0
    point : array-like
        The values for x, y and z for the point that you want
        to calculate the distance to.

    Returns
    -------
    float
        Distance from the point to the plane as defined in the
        References listed below.

    References
    ----------
    .. [1] Math Insight, Distance from point to plane,
           http://mathinsight.org/distance_point_plane
    .. [2] Ballantine, J. P., Essentials of Engineering Mathematics
           Prentice Hall, 1938.
    """
    abc = abcd[:3]
    d = abcd[3]

    dist = np.abs(np.dot(abc, point)+d)
    return dist/np.linalg.norm(abc)


def compute_coefficients(xyz):
    """Fit a plane to the first three dimensions of a matrix

    Parameters
    ----------
    xyz : array-like
        The matrix of data to fit the plane to.

    Returns
    -------
    np.array
        1-dimensional array with four values, the coefficients `a`, `b`, `c`
        and `d` in the equation:

    .. math::
        a\ x + b\ y - c\ z + d = 0.

    """
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    A = np.column_stack([x, y, np.ones_like(x)])
    abd, residuals, rank, s = np.linalg.lstsq(A, z)

    # add the coefficient of Z to
    return np.insert(abd, 2, -1)


def point_to_segment_distance(abcd, point, xyz):
    """Compute the distance from a point to a segment of a plane

    Parameters
    ----------
    point : array-like
        A point in space
    xyz : array-like
        Points that were used to define the plane
    abcd : array-like
        Coefficients of the plane in the form a*x + b*y + c*z + d = 0, where
        the equivalent array looks like [a, b, c, d]

    Returns
    -------
    float
        Distance from the point to the segment of the plane that spans
        throughout the points in `xyz`.

    Notes
    -----
    This function doesn't handle the N-dimensional problem and is specific to 3
    dimensions, but it should be straight-forward to extend into an
    N-dimensional solution.

    References
    ----------
    .. [1] http://stackoverflow.com/a/16459129
    """
    def plane(_abcd, xy):
        _a, _b, _c, _d = _abcd
        x, y = xy
        return (_a*x + _b*y + _d)/(-1*_c)

    a, b, c, d = abcd
    p, q, r = point
    l = ((d*-1.0) - p*a - b*q -c*r) / (a**2 +b**2 +c**2)
    extreme = np.array([p + l*a, q + l*b, r + l*c])

    for i in range(xyz.shape[-1]):
        vector = xyz[:, i]
        ranges = (vector.min(), vector.max())

        if extreme[i] < ranges[0]:
            extreme[i] = ranges[0]
        elif extreme[i] > ranges[1]:
            extreme[i] = ranges[1]

    extreme[-1] = plane(abcd, extreme[:-1])

    return euclidean(point, extreme)


def distance_to_reference_plane(ordination, metadata, category, column=None):
    """Compute all distances to a reference plane

    Parameters
    ----------
    ordination : skbio.OrdinationResults
        Ordinated coordinates with a reference set of coordinates.
    metadata : pd.DataFrame or pd.Series
        Pandas DataFrame with sample metadata, note it must be indexed by the
        sample identifier.
    category : str
        Category value that subsets the reference plane.
    column : str, optional
        If `metadata` is a DataFrame, the name of the column where `category`
        is present.

    Returns
    -------
    pd.Series
        Distances from every sample in `ordination` to the reference plane.

    Raises
    ------
    ValueError
        If no samples overlap between `ordination` and `metadata`.
        If both `column` is None and `metadata` is not a pd.Series object.
        If there are no samples found specified by `column` and/or `category`.

    Notes
    -----
    The samples represented in `ordination` should all be represented by a
    row in `metadata`.
    """

    # filter the metadata to only the data present in the ordination
    metadata = metadata.loc[ordination.samples.index].copy()

    if len(metadata) == 0:
        raise ValueError("There are no overlapping samples in your metadata, "
                         "and your ordinated coordinates.")

    if column is not None:
        reference_ids = metadata[metadata[column] == category].index
    elif column is None and isinstance(metadata, pd.Series):
        reference_ids = metadata[metadata == category].index
    else:
        raise ValueError("If `column` is `None`, `metadata` has to be a "
                         "pd.Series object.")

    if len(reference_ids) == 0:
        raise ValueError("Cannot find reference points using the provided "
                         "data")

    # slice the reference coordinates and only keep the first 3 dimensions
    reference = ordination.samples.loc[reference_ids].values[:, :3]

    abcd = compute_coefficients(reference)

    def funky(town):
        return point_to_segment_distance(abcd, town.values[:3], reference)

    return ordination.samples.apply(funky, axis=1, reduce=False)
