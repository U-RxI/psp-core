# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 09:14:56 2022

@author: Asmus Graumann Moser
"""

from cmath import cos, sin, pi, phase
from collections.abc import Iterable
from numpy import linspace
from shapely.geometry import LineString


def P2R(magnitude: float, angle_deg: float) -> complex:
    """Convert Polar coordinates (r, \\phi) to Rectangular form (x + 1j*y)

    Parameters
    ----------
    magnitude : float
        magnitude (r)
    angle_deg : float
        angle in degrees (\\phi)

    Returns
    -------
    complex
        Rectangular form (x + 1j*y)

    Example
    -------
    >>> print(P2R(1, 60))
    0.5000000000000001+0.8660254037844386j
    """

    angle_rad = angle_deg / 180 * pi
    return magnitude * (cos(angle_rad) + 1j * sin(angle_rad))

def angle_deg(z : complex) -> float:
    """Return phase angle in degrees

    Parameters
    ----------
    z : complex
        Complex number

    Returns
    -------
    float
        Phase angle (argument) in degrees

    Example
    -------
    from protlib import angle_deg
    
    angle_deg(1+1j) -> 45.0
    """
    
    return phase(z) / pi * 180

def rad2deg(ang : float) -> float:
    """Convert angle in radians to degree

    Parameters
    ----------
    ang : float
        Angle in radian.

    Returns
    -------
    float
        Angle in degree.
    
    Example
    -------
    >>> from cmath import pi
    >>> print(rad2deg(pi))
    180.0
    """
    
    return ang / pi * 180

def deg2rad(ang : float) -> float:
    """Convert angle in degree to radian

    Parameters
    ----------
    ang : float
        Angle in degree.

    Returns
    -------
    float
        Angle in radian.
    
    Example
    -------
    >>> print(deg2rad(180))
    3.141592653589793
    """
    return ang / 180 * pi

def _polar_str(z: complex, nd : int) -> str:
    """Convert a complex number to a string representing the polar coordinates.

    Parameters
    ----------
    z : complex
        Complex number or list of commplex numbers.
    nd : int, optional
        Number of decimals in the magnitude and angle. The default is 2.

    Returns
    -------
    str
        String with the polar coordinates. e.g. '1.41∠45.00°'

    Example
    -------
    from protlib import polar_str, P2R
    print( polar_str(1 + 1j) )
    print( polar_str(1 + 1j, nd = 1) )
    print( polar_str(P2R(1, 120)) )
    
    # output
    1.41∠45.00°
    1.4∠45.0°
    1.00∠120.00°
    
    """
    magnitude = abs(cround(z))
    angle = angle_deg(cround(z))
    if magnitude == 0:
        angle = 0
    return f"{magnitude:.{nd}f}\N{ANGLE}{angle:.{nd}f}\N{DEGREE SIGN}"

def polar_str(z: complex | Iterable[complex, ...], nd: int = 2) -> str | list[str, ...]:
    """Convert a complex number or list of complex numbers to a string or list
    of strings representing the polar coordinates.

    Parameters
    ----------
    z : complex | Iterable[complex, ...]
        Complex number or list of commplex numbers.
    nd : int, optional
        Number of decimals in the magnitude and angle. The default is 2.

    Returns
    -------
    str | list[str, ...]
        String or list with the polar coordinates. e.g. '1.41∠45.00°'

    Example
    -------
    from protlib import polar_str, P2R
    print( polar_str(1 + 1j) )
    print( polar_str(1 + 1j, nd = 1) )
    print( polar_str(P2R(1, 120)) )
    print( polar_str([P2R(1, 120), P2R(1, -120)]) )
    print( polar_str(P2R(1, 0) - P2R(1, -120) ) )
    
    # output
    1.41∠45.00°
    1.4∠45.0°
    1.00∠120.00°
    ['1.00∠120.00°', '1.00∠-120.00°']
    1.73∠30.00°
    
    """
    
    if isinstance(z, complex):

        return _polar_str(z, nd)
    
    elif isinstance(z, Iterable):
        result = []
        for cn in z:
            if not isinstance(cn, complex):
              raise ValueError("Iterable must contain complex only")
              
            result.append(_polar_str(cn, nd))
              
        return result

    else:
        raise ValueError("Input 'z' must be a complex number"
                         "or a iterble of complex numbers")

def symmetrical_components(
    A: complex, B: complex, C: complex
) -> tuple[complex, complex, complex]:
    """Calculates the symmetrical components for a 3 phase system.
    
    This done by applying the inverse A matrix:
        
        s_012 = A**(-1) * s_abc
        
        A**(-1) = 1/3 * [ [1, 1, 1],
                        [1, a, a**2],
                        [1, a**2, a] ]

    Parameters
    ----------
    A : complex
        Phase A / L1.
    B : complex
        Phase B / L2.
    C : complex
        Phase C / L3.

    Returns
    -------
    Tuple[complex, complex, complex]
        Returns a tuple with zero, positive and negative sequence.

    Example
    -------
    from protlib import P2R, symmetrical_components

    # Phase-Phase fault (BC)
    # Currents
    IA = P2R(0.40, 173.21)
    IB = P2R(0.40, -164.54)
    IC = P2R(0.79, 4.35)
    I0, I1, I2 = symmetrical_components( IA, IB, IC )
    print(polar_str(I0))
    print(polar_str(I1))
    print(polar_str(I2))

    # output
    0.00∠6.69°
    0.35∠-115.65°
    0.44∠124.34°
    
    # Phase-Phase fault (high impedance grounded)
    # Voltages
    UA = P2R(0.0, 0)
    UB = P2R(1.732, -150)
    UC = P2R(1.732, 150)
    U0, U1, U2 = symmetrical_components( UA, UB, UC )
    print(polar_str(3 * U0))
    print(polar_str(U1))
    print(polar_str(U2))
    
    # output
    3.00∠180.00°
    1.00∠-0.00°
    0.00∠0.00°

    """

    # 120 degree phase shift
    _a_ = P2R(1, 120)
    # 240 degree phase shift
    _a2_ = P2R(1, 240)
    # Zero sequence
    s0 = (1 / 3) * (A + B + C)
    # Positive sequence
    s1 = (1 / 3) * (A + _a_ * B + _a2_ * C)
    # Negative sequence
    s2 = (1 / 3) * (A + _a2_ * B + _a_ * C)

    return s0, s1, s2

def polarizing(X_A : complex, X_B : complex, X_C : complex) -> tuple[complex, complex, complex]:
    """Returning polarized quantities for a three phase system.
    Parameters
    ----------
    X_A : complex
        Phase A.
    X_B : complex
        Phase B.
    X_C : complex
        Phase C.

    Returns
    -------
    X_AB : complex
        DESCRIPTION.
    X_BC : complex
        DESCRIPTION.
    X_CA : complex
        DESCRIPTION.
    
    Example
    -------
    from protlib import P2R, polarizing
    
    V_AB, V_BC, V_CA = polarizing(P2R(1, 0), P2R(1, -120), P2R(1, 120))
    print(polar_str(V_AB))
    print(polar_str(V_BC))
    print(polar_str(V_CA))
    
    # output
    1.73∠30.00°
    1.73∠-90.00°
    1.73∠150.00°
    
    """
    
    X_AB = X_A - X_B
    X_BC = X_B - X_C
    X_CA = X_C - X_A
    
    return X_AB, X_BC, X_CA

# Need some more testing
def directional_generic(
        X_pol : complex, X_op : complex, RCA : float, ROA : float,
        phi_n : float, default : str, 
        ) -> tuple[complex, complex, complex, complex, str]:
    '''Generic implementation of a directional element. Suitable for phase,
    zero-sequence and negative sequence. 

    Parameters
    ----------
    X_pol : complex
        Polarizing quantity which is the reference for X_op.
        Usually a voltage like UA/UB/UC, U0 or U2
    X_op : complex
        Operating quantity. Usually a current IA/IB/IC, I0 or I2
    RCA : float
        Relay Charateristic Angle.
        This is the phase angle the polarizing quantity is rotated with.
        Also called Directional angle or MTA (maximum torque angle).
    ROA : float
        Relay Open Angle [degrees].
        The angle defines the directional area as RCA +/- ROA.
        Should not be more than 90 degrees. 
    phi_n : float
        Phase angle to normalize the polarizing quantity X_pol.
        E.g. if norm = -90 it places the phasor X_pol at the negative y-axis (-Im).
        All the other quantities are normalized by rotating with the phasor 1∠φ_n°.
         
    default : str
        Default refers to the default direction.
        It should be either 'Forward', 'Backward' or 'Undefined'.
        This it to support different implementation of directional elements.
        If ROA is less than 90 degress e.g. 85 degress,
        there are 2 x 10 degress of undefined areas.
        The relay decision can for the area can be set with this default value.

    Returns
    -------
    tuple[complex, complex, complex, complex, str]
        X_pol_n : complex
            The normalized polarizing quantity.
            |X_pol|∠norm°
        X_op_n : complex
            The normalized operating quantity with reference to RCA
            X_op * 1∠( -φ(X_pol) + φ_n - RCA )°
        a : complex
            Phasor to describe boundary of directional zone.
            a = b + 2*ROA
        b : complex
            Phasor to describe boundary of directional zone.
            b = a - 2*ROA

    '''
    
    # RCA leading or lagging? to be described
    
    X_pol_n = P2R(abs(X_pol), phi_n)
    
    # X_op rotated by -phi(X_pol), phi_n and -RCA
    X_op_n = X_op * P2R(1, - ( angle_deg(X_pol) - phi_n )  ) * P2R(1, - RCA)
    print(X_op_n)
    print(X_op * P2R(1, - angle_deg(X_pol)) * P2R(1, phi_n) * P2R(1, - RCA))
    print(X_op * P2R(1, - angle_deg(X_pol) + phi_n - RCA))
    
    # +-ROA rotaded with a normalzing phasor
    a = P2R(1, -ROA + phi_n)
    b = P2R(1, ROA + phi_n)
    
    direction = default
    
    if phase(a) < phase(X_op_n) < phase(b):
        direction = 'Forward'

    # X_op_n rotated 180 deg
    if phase(a) < phase(-X_op_n) < phase(b):
        direction = 'Backward'

    return X_pol_n, X_op_n, a, b, direction

def neg_dir_ABB(V_2 : complex, I_2 : complex, RCA : float, ROA : float ):
    V_pol_n, I_op_n, a, b, direction = directional_generic(X_pol = -V_2,
                                                           X_op  =  I_2,
                                                           RCA   = -RCA,
                                                           ROA   =  ROA,
                                                           phi_n  =  0,
                                                           default   = 'Undefined' )
    return V_pol_n, I_op_n, a, b, direction

def neg_dir_Siemens(V_2 : complex, I_2 : complex, alpha : float, beta : float ):
    ROA = (beta + (360-alpha) )/ 2
    RCA = beta-ROA # or alpha + ROA - 360
    I_pol_n, U_op_n, alpha, beta, direction = directional_generic(X_pol = -I_2,
                                                           X_op  =  V_2,
                                                           RCA   =  RCA,
                                                           ROA   =  ROA,
                                                           phi_n   =  0,
                                                           default   = 'Backward' )
    return U_op_n, I_pol_n, alpha, beta, direction

### Needs to be tested more
def pos_dir_generic(U_f: complex, I_f: complex, alpha: float = -15, beta: float = 115, U_pre: complex = 0 + 0j, k: float = 1.0) -> tuple[complex, float, str]:
    """Calculating reponse of directional element using positive sequence method

    Parameters
    ----------
    U_f : complex
        Voltage of the faulted phase [kV].
    I_f : complex
        Current of the faulted phase [kA].
    alpha : float, optional
        Boundary angle for forward direction [°]. The default is -15 degrees.
    beta : float, optional
        Boundary angle for forward direction. The default is 115 degrees.
    U_pre : complex, optional
        Pre-fault voltage. The default is 0 + 0j.
    k : float, optional
        Weight factor for weighing the pre-fault voltage and fault voltage.
        The default is 1.0, which weigh fault voltage 100% and pre-fault (1 - k) -> 0%

    Returns
    -------
    Tuple[complex, float, str]
        DESCRIPTION.

    """
    
    
    #U_A = P2R(38.48, -3)
    #I_A = P2R(1044, -40.85)
    #dir_positive_seq(U_f = U_A, I_f = I_A)
    
    U_pol = (1 - k) * U_pre + k * U_f
    
    arg = phase(U_pol / I_f) / pi * 180
    
    # normalized argument
    arg_norm = arg
    if arg < 0:
        arg_norm = 360 + arg
        
    direction = 'Undefined'
    if alpha < arg_norm < beta:
        direction = 'Forward'
        
    if (alpha + 180) < arg_norm < (beta + 180):
        direction = 'Backward'

    return U_pol, arg, direction

def reactance_method_pg(V_s: complex, I: complex , I_0: complex, Z_1L: complex, k : complex) -> float:
    # This is only intended for cables of same type.
    # Earth fault factor
    
    I_s = I + k * 3 * I_0
    m = (V_s / I_s).imag / Z_1L.imag
   
    return m

def modi_takagi(V_s, I, I0, I2, Z_1L, k):
    # source: 6654_TutorialFault_NF_20140312_Web.pdf
    
    I_s = I + 3 * I0 * k
    Iconj = I2.conjugate()
    m = (V_s*Iconj).imag/(Z_1L*I_s*Iconj).imag
    return m

def cround(x:complex, nd:int = 10):
    """ Function to round complex numbers.

    Parameters
    ----------
    x : complex
        Complex number to be rounded.
    nd : int, optional
        Number of digits. The default is 10.

    Returns
    -------
    complex
        Rounded complex number.
    
    Example
    -------
    from protlib import cround
    
    print(cround(1.12345 + 1.12345j, nd = 2))
    
    # output
    (1.12+1.12j)

    """
    return complex(round(x.real, nd),round(x.imag, nd))

def zeroseq2phase(U0 : complex, I0 : complex) -> tuple[complex, complex, complex, complex, complex, complex]:
    UL1_E = UL2_E = UL3_E = U0
    IL1 = IL2 = IL3 = I0
    return UL1_E, UL2_E, UL3_E, IL1, IL2, IL3

def negseq2phase(U2 : complex, I2 : complex) -> tuple[complex, complex, complex, complex, complex, complex]:
    UL1_E, UL2_E, UL3_E = sequence2phase(0, 0, U2)
    IL1, IL2, IL3 = sequence2phase(0, 0, I2)
    return UL1_E, UL2_E, UL3_E, IL1, IL2, IL3

def sequence2phase(s0: complex, s1: complex, s2: complex) -> tuple[complex, complex, complex]:
    '''Convert sequence componets back to phase components for a three phase
    system.
        
    This is function does the opposite of the symmetrical_component function,
    by applying the A matrix:
        
        s_abc = A * s_012
        
        A = [ [1, 1, 1],
             [1, a**2, a],
             [1, a, a**2] ]
    

    Parameters
    ----------
    s0 : complex
        Zero sequence component.
    s1 : complex
        Positive sequence component.
    s2 : complex
        Negative sequence component.

    Returns
    -------
    tuple(complex, complex, complex)
        sA : complex
            Phase voltage/current A.
        sB : complex
            Phase voltage/current B.
        sC : complex
            Phase voltage/current C.
    
    Example
    -------
    from protlib import sequence2phase, symmetrical_components, P2R, polar_str
    
    UA = P2R(0, 0)
    UB = P2R(1, -120)
    UC = P2R(1, 120)
    
    print("Phase voltage:")
    print(f"UA = {polar_str(UA)}, UB = {polar_str(UB)}, UC = {polar_str(UC)}")
    
    print("Decomposed to Zero, negative and positive components:")
    U0, U1, U2 = symmetrical_components(UA, UB, UC)
    
    print(f"U0 = {polar_str(U0)}, U1 = {polar_str(U1)}, U2 = {polar_str(U2)}")
    
    print("Sequence components composed back to phase voltages:")
    UA_, UB_, UC_ = sequence2phase(U0, U1, U2)
    
    print(f"UA = {polar_str(UA_)}, UB = {polar_str(UB_)},"
          f"UC = {polar_str(UC_)}")
    
    # output
    Phase voltage:
    UA = 0.00∠0.00°, UB = 1.00∠-120.00°, UC = 1.00∠120.00°
    Decomposed to Zero, negative and positive components:
    U0 = 0.33∠180.00°, U1 = 0.67∠-0.00°, U2 = 0.33∠180.00°
    Sequence components composed back to phase voltages:
    UA = 0.00∠0.00°, UB = 1.00∠-120.00°,UC = 1.00∠120.00°

    '''
    
    # 120 degree phase shift
    _a_ = P2R(1, 120)
    # 240 degree phase shift
    _a2_ = P2R(1, 240)
    # Phase A
    sA = s0 + s1 + s2
    # Phase B
    sB = s0 + _a2_ * s1 + _a_ * s2
    # Phase C
    sC = s0 + _a_ * s1 + _a2_ * s2
    
    return sA, sB, sC

#### Loop impedances

def Z_Ph_E_conv(U_Ph_E : complex, IL : complex, IE, kE : complex) -> complex:
    '''
    Function to calculate the phase-earth loop impedance as a complex number,
    using the traditional/conventional method (U / (IL - kE * IE)). This method
    assumes that Rf = 0 and IL & IE are exactly 180 degrees apart.
    Reference: Gerhard Ziegler - "Numerical distance protection"
    Note: Convention IL and IE is 180 degree apart

    Parameters
    ----------
    U_Ph_E : complex
        Phase-earth voltage represented as a complex number [V].
    IL : complex
        Phase current represented as a complex number [A].
    IE : complex
        Earth current represented as a complex number [A].
    kE : complex
        Residual compensation factor kE = (Z0 - Z1) / (3 * Z1).

    Returns
    -------
    Z_Ph_E : complex
        The phase-earth impedance loop [Ohm].

    '''
    
    Z_Ph_E = U_Ph_E / (IL - kE * IE)
    
    return Z_Ph_E

def Z_Ph_E(U_Ph_E : complex, IL : complex, IE : complex, RE_RL : float, XE_XL : float) -> complex:
    '''
    Function to calculate the phase-earth loop impedance as a complex number.
    Reference: Gerhard Ziegler - "Numerical distance protection" page 101
    Note: Convention IL and IE is ~180 degree apart
    
    Parameters
    ----------
    U_Ph_E : complex
        Phase-earth voltage represented as a complex number [V].
    IL : complex
        Phase current represented as a complex number [A].
    IE : complex
        Earth current represented as a complex number [A].
    RE_RL : float
        Resistive residual compensation factor RE_RL = (1/3) * (R0/R1 - 1).
    XE_XL : float
        Reactive residual compensation factor XE_XL = (1/3) * (X0/X1 - 1).

    Returns
    -------
    complex
        The phase-earth impedance loop [Ohm].

    '''
    R = R_Ph_E(U_Ph_E, IL, IE, RE_RL, XE_XL)
    
    X = X_Ph_E(U_Ph_E, IL, IE, RE_RL, XE_XL)
    
    return R + X*1j

def R_Ph_E(U_Ph_E : complex, IL : complex, IE : complex, RE_RL : float, XE_XL : float) -> float:
    '''
    Function to calculate the resistive part of the phase-earth loop impedance.
    Reference: Gerhard Ziegler - "Numerical distance protection" page 101 (3-48)
    Note: Convention IL and IE is ~180 degree apart
    
    Parameters
    ----------
    U_Ph_E : complex
        Phase-earth voltage represented as a complex number [V].
    IL : complex
        Phase current represented as a complex number [A].
    IE : complex
        Earth current represented as a complex number [A].
    RE_RL : float
        Resistive residual compensation factor RE_RL = (1/3) * (R0/R1 - 1).
    XE_XL : float
        Reactive residual compensation factor XE_XL = (1/3) * (X0/X1 - 1).

    Returns
    -------
    float
        Real/resistive part of the phase-earth impedance loop [Ohm].

    '''
    
    IR = IL - RE_RL * IE # (3-50)
    
    IX = IL - XE_XL * IE # (3-50)
    
    Rph_E = (U_Ph_E.real * IX.real + U_Ph_E.imag * IX.imag) \
        /(IR.real * IX.real + IR.imag * IX.imag) # (3-48)
   
    return Rph_E

def X_Ph_E(U_Ph_E : complex, IL : complex, IE : complex, RE_RL : float, XE_XL : float) -> float:
    '''
    Function to calculate the reactive part of the phase-earth loop impedance.
    Reference: Gerhard Ziegler - "Numerical distance protection" page 101 (3-49)
    Note: Convention IL and IE is ~180 degree apart
    
    Parameters
    ----------
    U_Ph_E : complex
        Phase-earth voltage represented as a complex number [V].
    IL : complex
        Phase current represented as a complex number [A].
    IE : complex
        Earth current represented as a complex number [A].
    RE_RL : float
        Resistive residual compensation factor RE_RL = (1/3) * (R0/R1 - 1).
    XE_XL : float
        Reactive residual compensation factor XE_XL = (1/3) * (X0/X1 - 1).

    Returns
    -------
    float
        Imaginary/reactive part of the phase-earth impedance loop [Ohm].

    '''
    
    IR = IL - RE_RL * IE # (3-50)
    
    IX = IL - XE_XL * IE # (3-50)
    
    Xph_E = (U_Ph_E.imag * IR.real - (U_Ph_E.real * IR.imag)) \
        /(IR.real * IX.real + IR.imag * IX.imag) # (3-49)
    return Xph_E

def Z_Ph(Ux, Uy, Ix, Iy):
    pass

def R_Ph():
    pass

def X_Ph():
    pass

def loop(UA : complex, UB : complex, UC : complex, IA : complex, IB : complex , IC : complex, RE_RL : float, XE_XL : float, IN=None):
    if IN:
        pass
    else:
        IN = IA + IB + IC
    
    Z_UA_E = Z_Ph_E(UA, IA, IN, RE_RL, XE_XL)
    Z_UB_E = Z_Ph_E(UB, IB, IN, RE_RL, XE_XL)
    Z_UC_E = Z_Ph_E(UC, IC, IN, RE_RL, XE_XL)
    
    Z_AB = (UA - UB) / (IA - IB)
    Z_BC = (UB - UC) / (IB - IC)
    Z_CA = (UC - UA) / (IC - IA)
    
    return Z_UA_E, Z_UB_E, Z_UC_E, Z_AB, Z_BC, Z_CA

def transfer_PQ(polygon, Un):
    t = linspace(0, 1.0, 1000)
    linestring = LineString(polygon.exterior.coords)
    out = list(map(lambda x: linestring.interpolate(x, normalized=True).coords, t))

    x2 = []
    y2 = []
    for p in out:
        if 0 in list(*p):
            continue
        xp = P(Un, *p) / 1e6
        xq = Q(Un, *p) / 1e6
        x2.append(xp)
        y2.append(xq)
    
    return x2, y2

def P(U, Z):
    return abs(U)**2 / abs(complex(*Z))**2 * Z[0]

def Q(U, Z):
    return abs(U)**2 / abs(complex(*Z))**2 * Z[1]