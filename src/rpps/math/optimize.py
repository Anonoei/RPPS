"""Nonlinear least squares"""
from types import SimpleNamespace

import numpy as np

def _jacobian(fun, x, f0):
    """Forward-difference Jacobian (len(f0), len(x))"""
    J = np.empty((len(f0), len(x)))
    for i in range(len(x)):
        h = 1.4901161193847656e-8*max(abs(x[i]), 1.0) # sqrt(eps), the usual step
        xi = x.copy(); xi[i] += h
        J[:, i] = (np.asarray(fun(xi), float) - f0)/h
    return J

def least_squares(fun, x0, jac=None, max_nfev=None, xtol=1e-10, ftol=1e-10):
    """Minimise 0.5 * sum(fun(x)**2) from `x0` (Levenberg-Marquardt with Marquardt scaling)

    fun(x) -> residual vector; jac(x) -> its Jacobian (rows: residuals, columns: parameters), or None for forward differences.
    Returns a namespace like scipy's: x, fun (residuals at x), jac (Jacobian at x), cost, nfev, success
    """
    x = np.atleast_1d(np.asarray(x0, float)).copy()
    max_nfev = max_nfev or 100*len(x)
    f = np.atleast_1d(np.asarray(fun(x), float)); nfev = 1
    J = np.atleast_2d(jac(x)) if jac else _jacobian(fun, x, f)
    cost = 0.5*f@f
    lam, success = 1e-3, False
    while nfev < max_nfev:
        A, g = J.T@J, J.T@f
        if np.linalg.norm(g, np.inf) < 1e-12*max(cost, 1e-300):
            success = True
            break
        improved = False
        while nfev < max_nfev:
            try:
                step = np.linalg.solve(A + lam*np.diag(np.maximum(np.diag(A), 1e-12)), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            fn = np.atleast_1d(np.asarray(fun(x + step), float)); nfev += 1
            cn = 0.5*fn@fn
            if np.isfinite(cn) and cn < cost:
                small = np.linalg.norm(step) <= xtol*(np.linalg.norm(x) + xtol) or cost - cn <= ftol*cost
                x, f, cost = x + step, fn, cn
                lam = max(lam/3, 1e-12)
                improved = True
                break
            lam *= 4
            if lam > 1e16:
                break
        if not improved:
            success = True # No step lowers the cost: a minimum to this precision
            break
        J = np.atleast_2d(jac(x)) if jac else _jacobian(fun, x, f)
        if small:
            success = True
            break
    return SimpleNamespace(x=x, fun=f, jac=J, cost=cost, nfev=nfev, success=success)
