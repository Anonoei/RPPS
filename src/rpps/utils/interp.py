from encodings import kz1048
import numpy as np

def linear(x, xp, yp):
    return np.interp(x, xp, yp)

def lagrange(x, xp, yp):
    n = len(xp)
    # def get_basis(X, j):
    #     b = [
    #         (X - xp[m]) / (xp[j] - xp[m])
    #         for m in range(n) if not m == j
    #     ]
    #     return np.prod(b, axis=0) * yp[j]
    # return np.sum([get_basis(x, j) for j in range(n)], axis=0)

    basis = np.zeros((len(x), len(x)), dtype=yp.dtype)
    for j in range(n):
        b = []
        for m in range(n):
            if m == j:
                continue
            b.append((x - xp[m]) / (xp[j] - xp[m]))
        basis[j,:] = np.prod(b, axis=0) * yp[j]
    return np.sum(basis, axis=0)

def spline(x, xp, yp):
    """
    q(x) = (1-t(x))y1 + t(x)y2 + t(x)(1-t(x))((1- t(x))a+t(x)b)
    t(x) = (x-x1)/(x2-x1)
    a = k1(x2-x1)-(y2-y1)
    b = -k2(x2-x1)+y2-y1
    [
        [ a11 a12 0   ] [k0] = [b1]
        [ a21 a22 a23 ] [k1] = [b2]
        [ 0   a32 a33 ] [k2] = [b3]
    ]
    a11 = 2/(x1-x0)
    a12 = 1/(x1-x0)
    a21 = 1/(x1-x0)
    a22 = 2*( 1/(x1-x0) + 1/(x2-x1) )
    a23 = 1/(x2-x1)
    a32 = 1/(x2-x1)
    a33 = 2/(x2-x1)
    b1 = 3*( (y1-y0)/(x1-x0)**2 )
    b2 = 3*( (y1-y0)/(x1-x0)**2 + ( (y2-y1)/(x2-x1**2) ) )
    b3 = 3*( (y2-y1)/(x2-x1)**2 )
    """
    xp = [-1, 0, 3]
    yp = [0.5, 0, 3]

    px = xp[0:3]
    py = yp[0:3]
    a11 = 2/(px[1]-px[0])
    a12 = 1/(px[1]-px[0])
    a21 = 1/(px[1]-px[0])
    a22 = 2*(1/(px[1]-px[0]) + 1/(px[2]-px[1]))
    a23 = 1/(px[2]-px[1])
    a32 = 2/(px[2]-px[1])
    a33 = 2/(px[2]-px[1])
    b1 = 3*( (py[1]-py[0])/(px[1]-px[0])**2 )
    b2 = 3*( (py[1]-py[0])/(px[1]-px[0])**2 + (py[2]-py[1])/(px[2]-px[1])**2 )
    b3 = 3*( (py[2]-py[1])/(px[2] - px[1])**2 )

    a_mat = [
        a11, a12, 0,
        a21, a22, a23,
        0,   a32, a33
    ]
    b_mat = [b1, b2, b3]
    k0 = np.matmul(a_mat[0:3], b_mat)
    k1 = np.matmul(a_mat[3:6], b_mat)
    k2 = np.matmul(a_mat[6:9], b_mat)
    print(f"k0: {k0}")
    print(f"k1: {k1}")
    print(f"k2: {k2}")
