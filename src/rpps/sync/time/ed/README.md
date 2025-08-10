# Timing Error Detectors

- Min = -0.5Tm
- Max = +0.5Tm

### Terms
- Ts = sample time
- Tm = symbol time
- ϵ = timing error
- ϵΔ = timing error estimate
- eD[m] = error detection for symbol m
- z(mTm + ϵΔ) = matched filter output @ mTm
- d(mTm + ϵΔ) = derivative of matched filter output @ mTm
- c(mTm + ϵΔ) = conjugate of z(mTm + ϵΔ)
- p(nTs) = RRC pulse shape
- r(nTs) = RC pulse shape
- m = middle symbol (mTm)
- e = early symbol (mTm - Tm/2)
- l = late symbol (mTm + Tm/2)
- a = known symbol value
- b = last symbol value

## Maximum Likelihood:
### Data aided
- eD[m] = z(mTm + ϵΔ) * d(mTm + ϵΔ)
  - ϵΔ < ϵ and z(mTm + ϵΔ) > 0: increase ϵΔ
  - ϵΔ < ϵ and z(mTm + ϵΔ) < 0: increase ϵΔ
  - ϵΔ > ϵ and z(mTm + ϵΔ) > 0: reduce ϵΔ
  - ϵΔ > ϵ and z(mTm + ϵΔ) < 0: reduce ϵΔ

## Early-Late
eD[m] = m * (l - e) > 0
- m sampled near peak
- e sampled near zero
- l sampled near zero

### Non-Data Aided
- eD[m] = z(mTm + ϵΔ){ z(l + ϵΔ) - z(e + ϵΔ) }
- { c(l + ϵΔ) - c(e + ϵΔ) }
### Decision-Directed
- eD[m] = b[mTm]{ z(l + ϵΔ) - z(e + ϵΔ) }
### Data aided
- eD[m] = a[m]{ z(l + ϵΔ) - z(e + ϵΔ) }

## Zero Crossings
eD[m] = m * (e - l) > 0
- m sampled near zero
- e sampled near peak
- l sampled near peak

### Non-Data aided (Gardner)
- eD[m] = z(e + ϵΔ){ z( (m-1)Tm + ϵΔ) - z(mTm + ϵΔ) }
- {c( (m-1)Tm + ϵΔ) - c(mTm + ϵΔ)}
### Decision-Directed
- eD[m] = z(e + ϵΔ){ b[m-1] - b[m] }
### Data aided
- eD[m] = z(e + ϵΔ){ a[m-1] - a[m] }

## Meuller and Muller
- p(+Tm - ϵΔ) - p(-Tm - ϵΔ) -> 0
### Decision-Directed
- eD[m] = b[m-1] * z(mTm + ϵΔ) - b[m] * z((m-1)Tm + ϵΔ)
### Data aided
- eD[m] = a[m-1] * z(mTm + ϵΔ) - a[m] * z((m-1)Tm + ϵΔ)
