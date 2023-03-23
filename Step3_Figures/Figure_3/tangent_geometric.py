import warnings
from math import sqrt, floor

import numpy as np
from scipy import linalg

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.covariance import LedoitWolf
from sklearn.utils import deprecated
# from .. import signal
from nilearn._utils.extmath import is_spd


def check_square(matrix):
    """Raise a ValueError if the input matrix is square.

    Parameters
    ----------
    matrix : numpy.ndarray
        Input array.

    """
    if matrix.ndim != 2 or (matrix.shape[0] != matrix.shape[-1]):
        raise ValueError('Expected a square matrix, got array of shape'
                         ' {0}.'.format(matrix.shape))


def check_spd(matrix):
    """Raise a ValueError if the input matrix is not symmetric positive
    definite.

    Parameters
    ----------
    matrix : numpy.ndarray
        Input array.

    """
    if not is_spd(matrix, decimal=7):
        raise ValueError('Expected a symmetric positive definite matrix.')


def form_symmetric(function, eigenvalues, eigenvectors):
    """Return the symmetric matrix with the given eigenvectors and
    eigenvalues transformed by function.

    Parameters
    ----------
    function : function numpy.ndarray -> numpy.ndarray
        The transform to apply to the eigenvalues.

    eigenvalues : numpy.ndarray, shape (n_features, )
        Input argument of the function.

    eigenvectors : numpy.ndarray, shape (n_features, n_features)
        Unitary matrix.

    Returns
    -------
    output : numpy.ndarray, shape (n_features, n_features)
        The symmetric matrix obtained after transforming the eigenvalues, while
        keeping the same eigenvectors.

    """
    return np.dot(eigenvectors * function(eigenvalues), eigenvectors.T)


def map_eigenvalues(function, symmetric):
    """Matrix function, for real symmetric matrices. The function is applied
    to the eigenvalues of symmetric.

    Parameters
    ----------
    function : function numpy.ndarray -> numpy.ndarray
        The transform to apply to the eigenvalues.

    symmetric : numpy.ndarray, shape (n_features, n_features)
        The input symmetric matrix.

    Returns
    -------
    output : numpy.ndarray, shape (n_features, n_features)
        The new symmetric matrix obtained after transforming the eigenvalues,
        while keeping the same eigenvectors.

    Notes
    -----
    If input matrix is not real symmetric, no error is reported but result will
    be wrong.

    """
    eigenvalues, eigenvectors = linalg.eigh(symmetric)
    return form_symmetric(function, eigenvalues, eigenvectors)


def geometric_mean(matrices, init=None, max_iter=10, tol=1e-7):
    """Compute the geometric mean of symmetric positive definite matrices.

    The geometric mean of n positive definite matrices
    M_1, ..., M_n is the minimizer of the sum of squared distances from an
    arbitrary matrix to each input matrix M_k

    gmean(M_1, ..., M_n) = argmin_X sum_{k=1}^N dist(X, M_k)^2

    where the used distance is related to matrices logarithm

    dist(X, M_k) = ||log(X^{-1/2} M_k X^{-1/2)}||

    In case of positive numbers, this mean is the usual geometric mean.

    See Algorithm 3 of [1]_.

    References
    ----------
    .. [1] P. Thomas Fletcher, Sarang Joshi. Riemannian Geometry for the
       Statistical Analysis of Diffusion Tensor Data. Signal Processing, 2007.

    Parameters
    ----------
    matrices : list of numpy.ndarray, all of shape (n_features, n_features)
        List of matrices whose geometric mean to compute. Raise an error if the
        matrices are not all symmetric positive definite of the same shape.

    init : numpy.ndarray, shape (n_features, n_features), optional
        Initialization matrix, default to the arithmetic mean of matrices.
        Raise an error if the matrix is not symmetric positive definite of the
        same shape as the elements of matrices.

    max_iter : int, optional
        Maximal number of iterations. Default=10.

    tol : positive float or None, optional
        The tolerance to declare convergence: if the gradient norm goes below
        this value, the gradient descent is stopped. If None, no  check is
        performed. Default=1e-7.

    Returns
    -------
    gmean : numpy.ndarray, shape (n_features, n_features)
        Geometric mean of the matrices.

    """
    # Shape and symmetry positive definiteness checks
    n_features = matrices[0].shape[0]
    for matrix in matrices:
        check_square(matrix)
        if matrix.shape[0] != n_features:
            raise ValueError("Matrices are not of the same shape.")
        check_spd(matrix)

    # Initialization
    matrices = np.array(matrices)
    if init is None:
        gmean = np.mean(matrices, axis=0)
    else:
        check_square(init)
        if init.shape[0] != n_features:
            raise ValueError("Initialization has incorrect shape.")
        check_spd(init)
        gmean = init

    norm_old = np.inf
    step = 1.

    # Gradient descent
    for n in range(max_iter):
        # Computation of the gradient
        vals_gmean, vecs_gmean = linalg.eigh(gmean)
        gmean_inv_sqrt = form_symmetric(np.sqrt, 1. / vals_gmean, vecs_gmean)
        whitened_matrices = [gmean_inv_sqrt.dot(matrix).dot(gmean_inv_sqrt)
                             for matrix in matrices]
        logs = [map_eigenvalues(np.log, w_mat) for w_mat in whitened_matrices]
        logs_mean = np.mean(logs, axis=0)  # Covariant derivative is
        # - gmean.dot(logms_mean)
        if np.any(np.isnan(logs_mean)):
            raise FloatingPointError("Nan value after logarithm operation.")

        norm = np.linalg.norm(logs_mean)  # Norm of the covariant derivative on
        # the tangent space at point gmean

        # Update of the minimizer
        vals_log, vecs_log = linalg.eigh(logs_mean)
        gmean_sqrt = form_symmetric(np.sqrt, vals_gmean, vecs_gmean)
        # Move along the geodesic
        gmean = gmean_sqrt.dot(
            form_symmetric(np.exp, vals_log * step, vecs_log)).dot(gmean_sqrt)

        # Update the norm and the step size
        if norm < norm_old:
            norm_old = norm
        elif norm > norm_old:
            step = step / 2.
            norm = norm_old
        if tol is not None and norm / gmean.size < tol:
            break
    if tol is not None and norm / gmean.size >= tol:
        warnings.warn("Maximum number of iterations {0} reached without "
                      "getting to the requested tolerance level "
                      "{1}.".format(max_iter, tol))

    return gmean


def sym_matrix_to_vec(symmetric, discard_diagonal=False):
    """Return the flattened lower triangular part of an array.
    If diagonal is kept, diagonal elements are divided by sqrt(2) to conserve
    the norm.
    Acts on the last two dimensions of the array if not 2-dimensional.
    .. versionadded:: 0.3
    Parameters
    ----------
    symmetric : numpy.ndarray or list of numpy arrays, shape\
        (..., n_features, n_features)
        Input array.
    discard_diagonal : boolean, optional
        If True, the values of the diagonal are not returned.
        Default=False.
    Returns
    -------
    output : numpy.ndarray
        The output flattened lower triangular part of symmetric. Shape is
        (..., n_features * (n_features + 1) / 2) if discard_diagonal is False
        and (..., (n_features - 1) * n_features / 2) otherwise.
    """
    if discard_diagonal:
        # No scaling, we directly return the values
        tril_mask = np.tril(np.ones(symmetric.shape[-2:]), k=-1).astype(
            bool)
        return symmetric[..., tril_mask]
    scaling = np.ones(symmetric.shape[-2:])
    np.fill_diagonal(scaling, sqrt(2.))
    tril_mask = np.tril(np.ones(symmetric.shape[-2:])).astype(bool)
    return symmetric[..., tril_mask] / scaling[tril_mask]


def vec_to_sym_matrix(vec, diagonal=None):
    """Return the symmetric matrix given its flattened lower triangular part.
    Acts on the last dimension of the array if not 1-dimensional.
    Diagonal can be encompassed in vec or given separately. In both cases, note
    that diagonal elements are multiplied by sqrt(2).
    .. versionadded:: 0.3
    Parameters
    ----------
    vec : numpy.ndarray or list of numpy arrays, shape \
        (..., n_columns * (n_columns + 1) /2) or
        (..., (n_columns - 1) * n_columns / 2) if diagonal is given seperately.
        The input array.
    diagonal : numpy.ndarray, shape (..., n_columns), optional
        The diagonal array to be stacked to vec. If None, the diagonal is
        assumed to be included in vec.
    Returns
    -------
    sym : numpy.ndarray, shape (..., n_columns, n_columns).
        The output symmetric matrix.
    Notes
    -----
    This function is meant to be the inverse of sym_matrix_to_vec. If you have
    discarded the diagonal in sym_matrix_to_vec, you need to provide it
    separately to reconstruct the symmetric matrix. For instance this can be
    useful for correlation matrices for which we know the diagonal is 1.
    See also
    --------
    nilearn.connectome.sym_matrix_to_vec
    """
    n = vec.shape[-1]
    # Compute the number of the symmetric matrix columns
    # solve n_columns * (n_columns + 1) / 2 = n subject to n_columns > 0
    n_columns = (sqrt(8 * n + 1) - 1.) / 2
    if diagonal is not None:
        n_columns += 1

    if n_columns > floor(n_columns):
        raise ValueError(
            "Vector of unsuitable shape {0} can not be transformed to "
            "a symmetric matrix.".format(vec.shape))

    n_columns = int(n_columns)
    first_shape = vec.shape[:-1]
    if diagonal is not None:
        if diagonal.shape[:-1] != first_shape or\
                diagonal.shape[-1] != n_columns:
            raise ValueError("diagonal of shape {0} incompatible with vector "
                             "of shape {1}".format(diagonal.shape, vec.shape))

    sym = np.zeros(first_shape + (n_columns, n_columns))

    # Fill lower triangular part
    skip_diagonal = (diagonal is not None)
    mask = np.tril(np.ones((n_columns, n_columns)), k=-skip_diagonal).astype(
        bool)
    sym[..., mask] = vec

    # Fill upper triangular part
    sym.swapaxes(-1, -2)[..., mask] = vec

    # (Fill and) rescale diagonal terms
    mask.fill(False)
    np.fill_diagonal(mask, True)
    if diagonal is not None:
        sym[..., mask] = diagonal

    sym[..., mask] *= sqrt(2)

    return sym
