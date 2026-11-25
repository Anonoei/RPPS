"""Running stages over a signal cut into blocks"""
import numpy as np

import rpps as rp

def run(make, x, blocks, **rates):
    """x cut into `blocks`, each as a block of one stream, through a fresh stage"""
    stage = make()
    stream, outs, i = None, [], 0
    for b in blocks:
        d = rp.Data(x[i:i+b], stream=stream, **rates)
        stream = d.stream
        outs.append((d + stage).data)
        i += b
    return np.concatenate(outs)

def cuts(n):
    return ([n], [1000]*(n//1000), [777]*(n//777) + [n % 777], [1]*50 + [n-50])
