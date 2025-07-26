"""Meuller & Muller TED"""

def mm_dd(cur, prv, c_est, p_est):
    """
    cur, (Tm)  : current symbol
    prv: (-Tm) : previous symbol
    c_est: current symbol estimate
    p_est: previous symbol estimate
    """
    return p_est*cur - c_est*prv
