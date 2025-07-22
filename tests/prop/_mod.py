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
    if points == 2:
        # 0 1
        x = np.array([])
        for bit in bits:
            pulse = np.zeros(sps)
            pulse[0] = bit*2-1
            x = np.concatenate((x, pulse))
        return x
    elif points == 4:
        # 0 1
        # 2 3
        x = np.array([], dtype=np.complex64)
        for bit in bits:
            pulseI = np.zeros(sps)
            pulseQ = np.zeros(sps)
            if bit % 2 == 0:
                pulseI[0] = -1
                pulseQ[0] = -(bit-1)
            else:
                pulseI[0] = 1
                pulseQ[0] = -(bit-2)
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
    return np.convolve(samps, h, mode="same") # apply filter

def offset(samps, offset, Ts):
    t = np.arange(0, Ts*len(samps), Ts) # create time vector
    return samps * np.exp(1j*2*np.pi*offset*t) # perform freq shift

def rrc(taps, sps, beta=0.35):
    t = np.arange(-taps//2,0)/sps

    sin_arg = np.sin(np.pi*t*(1-beta))
    cos_arg = 4*beta*t*np.cos(np.pi*t*(1+beta))
    denom = np.pi*t*(1-(4*beta*t)**2)
    h = (sin_arg+cos_arg)/denom

    h = np.concat((h, [1+beta*(4/np.pi -1)], h[::-1]))
    return h
