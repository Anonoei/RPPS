# Modulation

## Quadrature
 - $I: I\cos(2 \pi ft)$
 - $Q: Q\sin(2 \pi ft)$
 - $x(t) = I\cos(2 \pi ft) + Q\sin(2 \pi ft)$

### Carrier
 - $A*\cos(2 \pi ft + \phi)$
    - $A$ is amplitude
    - $f$ is frequency
    - $\phi$ is phase
 - $( \sqrt{ I^2+Q^2 } ) \cos( 2 \pi ft + \tan^{-1}(\frac{Q}{I}) )$
 - $A = \sqrt{ I^2+Q^2 } )$
 - $\phi = \tan^{-1}(\frac{Q}{I})$
 - "carrier" is $\cos$

## BPSK

### General Form
$S_n(t)=\sqrt{ \frac{2 E_b}{T_b} } \cos(2 \pi ft + \pi(1-n)), n = 0,1$

This yields two phases, 0 and $\pi$ (radians); or $0\degree$ and $180\degree$

 0. $S_0(t)=-\sqrt{ \frac{2 E_b}{T_b} } \cos(2 \pi ft)$
 1. $S_1(t)= \sqrt{ \frac{2 E_b}{T_b} } \cos(2 \pi ft)$

 - $f$ is the frequency of the base band

### Basis function
$\phi(t) = \sqrt{ \frac{2}{T_b} } \cos(2 \pi ft)$

 0. $-\sqrt{ E_b } \phi(t)$
 1. $\sqrt{ T_b } \phi$

#### Bit error rate
 - $P_b = Q(\sqrt{ \frac{2E_b}{N_0} })$
 - $P_e = \frac{1}{2} \text{erfc} (\sqrt{ \frac{E_b}{N_0} })$

## QPSK

$S_n(t) = \sqrt{ \frac{2E_s}{T_s} } \cos(2 \pi f_c t + (2n - 1)\frac{4}{\pi}), n=1,2,3,4$

This yields four phases, $\pi/4$, $3\pi/4$, $5\pi/4$ and $7\pi/4$ (radians); or $45\degree$, $135\degree$, $225\degree$, $315\degree$

### Basis functions
 - $\phi_1(t) = \sqrt{ \frac{2}{T_s} } \cos(2 \pi f_c t)$ = $I$ or in-phase
 - $\phi_2(t) = \sqrt{ \frac{2}{T_s} } \sin(2 \pi f_c t)$ = $Q$ or quadrature
 - $(\plusmn \sqrt{ \frac{E_s}{2} } \plusmn \sqrt{ \frac{E_s}{2} })$

#### Bit error rate
 - $P_b = Q( \sqrt{ \frac{2 E_b}{N_0} } )$
 - $P_s = 1 - (1 - P_b)^2$
   - $= 2Q ( \sqrt{ \frac{E_s}{N_0} } ) - [Q(\sqrt{ \frac{E_s}{N_0} }) ]^2$
 - $P_s \approx 2Q(\sqrt{\frac{E_s}{N_0}})$
   - $= \text{erfc}(\sqrt{ \frac{E_s}{2 N_0} })$
   - $= \text{erfc}(\sqrt{ \frac{E_b}{N_0} })$
