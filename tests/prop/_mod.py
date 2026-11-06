import numpy as np

def get_bits(num, points):
    return np.random.randint(0, points, num)

def mod_bits(bits, points):
    if points == 2:
        deg = bits*360/2.0 + 180
    elif points == 4:
        deg = bits*360/4.0 + 45
    elif points == 8:
        deg = bits*360/8.0 + 22.5
    x_rad = deg*np.pi/180.0
    return np.cos(x_rad) + 1j*np.sin(x_rad)

def map_bits(bits, points, sps):
    iLUT = None
    qLUT = None
    if points == 2:
        # 0 1
        iLUT = [-1, 1]
        qLUT = [0, 0]
    elif points == 4:
        # 0 1
        # 2 3
        iLUT = [-1, 1, -1, 1]
        qLUT = [1, 1, -1, -1]
    if iLUT is None or qLUT is None:
        return None
    x = np.array([], dtype=np.complex64)
    for bit in bits:
        pulseI = np.zeros(sps, dtype=np.float32)
        pulseQ = np.zeros(sps, dtype=np.float32)
        pulseI[0] = iLUT[bit]
        pulseQ[0] = qLUT[bit]
        x = np.concatenate((x, pulseI + 1j*pulseQ))
    return x

def awgn(samps, power=0.1):
    _awgn = (np.random.randn(len(samps)) + 1j*np.random.randn(len(samps)))/np.sqrt(2)
    return samps + _awgn*np.sqrt(power)

def phase_noise(samps, power=0.1):
    _noise = np.random.randn(len(samps)) * power
    return samps * np.exp(1j*_noise)

def delay(samps, offset: 0.4):
    N = 21 # number of taps
    n = np.arange(-N//2, N//2)
    h = np.sinc(n - offset) # calc filter taps
    h *= np.hamming(N) # window the filter to make sure it decays to 0 on both sides
    h /= np.sum(h) # normalize to get unity gain, we don't want to change the amplitude/power
    return np.convolve(samps, h, mode="same").astype(np.complex64) # apply filter

def offset(samps, offset, Ts):
    t = np.arange(0, Ts*len(samps), Ts) # create time vector
    return (samps * np.exp(1j*2*np.pi*offset*t)).astype(np.complex64) # perform freq shift

def rrc(taps, sps, beta=0.35):
    # t = np.arange(-taps//2,taps//2)
    # i0 = t==0
    # i1 = (t==(sps/(4*beta))) | (t==-(sps/(4*beta)))
    # i2 = ~(i0 | i1)

    # h = np.zeros(taps)
    # h[i0] = 1/sps*(1+beta*(4/np.pi - 1))
    # h[i1] = beta/(sps*np.sqrt(2)) * (
    #             (1+2/np.pi) * np.sin(np.pi/(4*beta)) + \
    #             (1-2/np.pi) * np.cos(np.pi/(4*beta))
    #         )
    # h[i2] = 1/sps*((
    #             np.sin(np.pi*t[i2]/sps * (1-beta)) \
    #             + 4*beta*t[i2]/sps * \
    #             np.cos(np.pi*t[i2]/sps * (1+beta))) / \
    #             (np.pi*t[i2]/sps* (1 - (4*beta*t[i2]/sps)**2))
    #         )
    t = np.arange(-taps//2,0)/sps # only generate half the pulse
    sin_arg = np.sin(np.pi*t*(1-beta))
    cos_arg = 4*beta*t*np.cos(np.pi*t*(1+beta))
    denom = np.pi*t*(1-(4*beta*t)**2)
    h = 1/sps*(sin_arg+cos_arg)/denom

    h = np.concat((h, [1/sps*(1+beta*(4/np.pi -1))], h[::-1]))
    h = h * np.hanning(len(h))
    hc = np.zeros(len(h), dtype=np.complex64)
    hc = h+1j*h
    return hc.astype(np.complex64)

def low_pass(taps, cut_off):
    t = np.arange(-taps//2+1,0)/cut_off
    h = np.sinc(t)
    h = np.concat((h, [np.sinc(0)], h[::-1]))
    h = h * np.hamming(taps)
    h = h/cut_off
    return h

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=a.dtype)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:]/n
