import scipy.special as sc
from scipy import io
import numpy as np
import math

def Fluspect(spectral,leafbio,optipar):
    """
    Calculates reflectance and transmittance spectra of a leaf using FLUSPECT, 
    plus four excitation-fluorescence matrices

    Based on the original algorithm by: 
    Wout Verhoef, Christiaan van der Tol (c.vandertol@utwente.nl), 
    Joris Timmermans, Nastassia Vilfan (2007-2020)

    Update from PROSPECT to FLUSPECT: January 2011 (CvdT)

    Python code by: G.Ntakos
    Date: 11/2022

    for more information:
    https://scope-model.readthedocs.io/en/master/api/RTMs.html#src.RTMs.fluspect_B_CX
    """

    # Fluspect parameters
    ndub        = 15          # number of doublings applied
    intst       = 5
    Cab         = leafbio["Cab"].item()
    Cca         = leafbio["Cca"].item()
    V2Z         = leafbio["V2Z"].item()
    Cw          = leafbio["Cw"].item()
    Cdm         = leafbio["Cdm"].item()
    Cs          = leafbio["Cs"].item()
    Cant 	    = leafbio["Cant"].item()
    Cbc         = leafbio["Cbc"].item()
    Cp          = leafbio["Cp"].item()
    N           = leafbio["N"].item()
    fqe         = leafbio["fqe"].item()

    nr          = optipar["nr"]
    Kdm         = optipar["Kdm"]
    Kab         = optipar["Kab"]

    if V2Z      == -999:
    # Use old Kca spectrum if this is given as input
        Kca     = optipar['Kca']
    else:
    # Otherwise make linear combination based on V2Z
    # For V2Z going from 0 to 1 we go from Viola to Zea
        Kca     = (1-V2Z) * optipar["KcaV"] + V2Z * optipar["KcaZ"]

    Kw          = optipar['Kw']
    Ks          = optipar['Ks']
    Kant        = optipar['Kant']
    
    if 'Kp' in optipar:
        Kp      = optipar['Kp']
    else:
        Kp      = 0*Kab
    
    if 'Kcbc' in optipar:
        Kcbc    = optipar['Kcbc']
    else:
        Kcbc    = 0*Kab
    
    phi         = optipar['phi']

    #PROSPECT calculations
    Kall        = (Cab*Kab + Cca*Kca + Cdm*Kdm + Cw*Kw  + Cs*Ks + Cant*Kant + Cp*Kp + Cbc*Kcbc)/N
    j           = Kall > 0
    t1          = (1-Kall) * np.exp(-Kall)
    t2          = Kall**2 * sc.exp1(Kall)
    tau         = np.ones_like(t1)
    tau[j]      = t1[j] + t2[j]
    kCarrel     = np.zeros((tau.size,1))
    kChlrel     = np.zeros((tau.size,1))
    kChlrel[j]  = (Cab*Kab[j])/(Kall[j]*N)
    kCarrel[j]  = (Cca*Kca[j])/(Kall[j]*N)
    talf        = calctav(59,nr)
    ralf        = 1-talf
    t12         = calctav(90,nr)
    r12         = 1-t12
    t21         = t12/(nr**2)
    r21         = 1-t21
    
    # top surface side
    denom       = 1-r21*r21*tau**2
    Ta          = talf*tau*t21/denom
    Ra          = ralf+r21*tau*Ta

    # bottom surface side
    t           = t12*tau*t21/denom
    r           = r12+r21*tau*t

    # Stokes equations to compute properties of next N-1 layers (N real)
    # Normal case
    D           = np.sqrt((1+r+t)*(1+r-t)*(1-r+t)*(1-r-t))
    rq          = r**2
    tq          = t**2
    a           = (1+rq-tq+D)/(2*r)
    b           = (1-rq+tq+D)/(2*t)
    bNm1        = b**(N-1)
    bN2         = bNm1**2
    a2          = a**2
    denom       = a2*bN2-1
    Rsub        = a*(bN2-1)/denom
    Tsub        = bNm1*(a2-1)/denom

    #			Case of zero absorption
    j           = r + t >= 1.0
    Tsub[j]     = t[j]/(t[j]+(1-t[j])*(N-1))
    Rsub[j]	    = 1-Tsub[j]

    # Reflectance and transmittance of the leaf: combine top layer with next N-1 layers
    denom       = 1-Rsub*r
    tran        = Ta*Tsub/denom
    refl        = Ra+Ta*Rsub*t/denom

    # Create output dictionary
    leafopt     = {
        "refl": refl,
        "tran": tran,
        "kChlrel": kChlrel,
        "kCarrel": kCarrel
    }
    
    # From here a new path is taken: The doubling method used to calculate
    # fluoresence is now only applied to the part of the leaf where absorption
    # takes place, that is, the part exclusive of the leaf-air interfaces. The
    # reflectance (rho) and transmittance (tau) of this part of the leaf are
    # now determined by "subtracting" the interfaces

    Rb          = (refl-ralf)/(talf*t21+(refl-ralf)*r21)  # Remove the top interface
    Z           = tran*(1-Rb*r21)/(talf*t21)             # Derive Z from the transmittance

    rho         = (Rb-r21*Z**2)/(1-(r21*Z)**2)    # Reflectance and transmittance 
    tau         = (1-Rb*r21)/(1-(r21*Z)**2)*Z    # of the leaf mesophyll layer
    t           = tau
    r           = np.maximum(rho,0)                       # Avoid negative r

    # Derive Kubelka-Munk s and k
    I_rt        = (r+t)<1

    D[I_rt]     = np.round(np.sqrt((1 + r[I_rt] + t[I_rt]) * \
                (1 + r[I_rt] - t[I_rt]) * \
                (1 - r[I_rt] + t[I_rt]) * \
                (1 - r[I_rt] - t[I_rt])),4)
    a[I_rt]     = (1 + r[I_rt]**2 - t[I_rt]**2 + D[I_rt]) / (2*r[I_rt])
    b[I_rt]     = (1 - r[I_rt]**2 + t[I_rt]**2 + D[I_rt]) / (2*t[I_rt])
    a[(~I_rt.astype('bool'))] =   1
    b[(~I_rt.astype('bool'))] =   1
    s           = r/t
    I_a         = ((a>1) & (a!=math.inf))
    s[I_a]      = 2.*a[I_a] / (a[I_a]**2 - 1) * np.log(b[I_a])
    k           = np.log(b)
    k[I_a]      = (a[I_a]-1) / (a[I_a]+1) * np.log(b[I_a])
    kChl        = kChlrel * k

    ## Fluorescence of the leaf mesophyll layer
    # Fluorescence part is skipped for fqe = 0
    #light version. The spectral resolution of the irradiance is lowered.
    if fqe      > 0:
        wle         = np.arange(400,755,intst).T.reshape(-1,1)
        k_iwle      = np.interp(wle.flatten(), spectral["wlP"].flatten(),k.flatten()).reshape(-1,1)
        s_iwle      = np.interp(wle.flatten(), spectral["wlP"].flatten(), s.flatten()).reshape(-1,1)
        kChl_iwle   = np.interp(wle.flatten(), spectral["wlP"].flatten(), kChl.flatten()).reshape(-1,1)
        r21_iwle    = np.interp(wle.flatten(),spectral["wlP"].flatten(), r21.flatten()).reshape(-1,1)
        rho_iwle    = np.interp(wle.flatten(),spectral["wlP"].flatten(), rho.flatten()).reshape(-1,1)
        tau_iwle    = np.interp(wle.flatten(),spectral["wlP"].flatten(), tau.flatten()).reshape(-1,1)
        talf_iwle   = np.interp(wle.flatten(),spectral["wlP"].flatten(), talf.flatten()).reshape(-1,1)
        fqe         = np.array([fqe])
        wlf         = np.arange(640,850,4).T.reshape(-1,1)
        wlp         = spectral["wlP"]
        Iwlf      = np.intersect1d(wlp,wlf, return_indices=True)[1]
        eps         = 2**(-ndub)

        # initialisations
        te          = 1-(k_iwle+s_iwle) * eps   
        tf          = 1-(k[Iwlf]+s[Iwlf]) * eps  
        re          = s_iwle * eps
        rf          = s[Iwlf] * eps
        sigmoid     = 1/(1+np.exp(-wlf/10)*np.exp(wle.T/10))
        Mf          = intst*fqe[0] * ((.5*phi[Iwlf])*eps) * kChl_iwle.T*sigmoid
        Mb          = intst*fqe[0] * ((.5*phi[Iwlf])*eps) * kChl_iwle.T*sigmoid
        Ih          = np.ones(len(te)).reshape(1,-1)
        Iv          = np.ones(len(tf)).reshape(-1,1)

        # Doubling routine
        for i in range(ndub):
            xe = te/(1-re*re)
            ten = te*xe
            ren = re*(1+ten)  
            xf = tf/(1-rf*rf)
            tfn = tf*xf
            rfn = rf*(1+tfn)
            A11  = xf*Ih + Iv*xe.T
            A12 = (xf*xe.T)*(rf*Ih + Iv*re.T)
            A21  = 1+(xf*xe.T)*(1+rf*re.T)
            A22 = (xf*rf)*Ih+Iv*(xe*re).T
            Mfn   = Mf  * A11 + Mb  * A12
            Mbn   = Mb  * A21 + Mf  * A22
            te   = ten
            re  = ren
            tf   = tfn
            rf   = rfn
            Mf  = Mfn
            Mb = Mbn

        # Here we add the leaf-air interfaces again for obtaining the final 
        # leaf level fluorescences.
        g = Mb
        f = Mf
        Rb = rho + tau**2*r21/(1-rho*r21)
        Rb_iwle = np.interp(wle.flatten(), spectral["wlP"].flatten(),Rb.flatten()).reshape(-1,1)
        
        Xe = Iv * (talf_iwle/(1-r21_iwle*Rb_iwle)).T
        Xf = t21[Iwlf]/(1-r21[Iwlf]*Rb[Iwlf]) * Ih
        Ye = Iv * (tau_iwle*r21_iwle/(1-rho_iwle*r21_iwle)).T
        Yf = tau[Iwlf]*r21[Iwlf]/(1-rho[Iwlf]*r21[Iwlf]) * Ih
    
        A = Xe * (1 + Ye*Yf) * Xf
        B = Xe * (Ye + Yf) * Xf
    
        gn = A * g + B * f
        fn = A * f + B * g
    
        leafopt["Mb"]  = gn
        leafopt["Mf"]  = fn
    return leafopt

def calctav(alfa,nr):

    rd          = math.pi/180
    n2          = nr**2
    nupi          = n2+1
    nm          = n2-1
    a           = (nr+1)*(nr+1)/2
    k           = -(n2-1)*(n2-1)/4
    sa          = math.sin(alfa*rd)

    b1          = int((alfa!=90))*np.sqrt(np.round((sa**2-nupi/2)**2+k,6))
    b2          = sa**2-nupi/2
    b           = b1-b2
    b3          = b**3
    a3          = a**3
    ts          = (k**2./(6*b3)+k/b-b/2)-(k**2/(6*a3)+k/a-a/2)

    tp1         = -2*n2*(b-a)/(nupi**2)
    tp2         = -2*n2*nupi*np.log(b/a)/(nm**2)
    tp3         = n2*(1/b-1/a)/2
    tp4         = 16*n2**2*(n2**2+1)*np.log((2*nupi*b-nm**2)/(2*nupi*a-nm**2))/(nupi**3*nm**2)
    tp5         = 16*n2**3*(1/(2*nupi*b-nm**2)-1/(2*nupi*a-nm**2))/(nupi**3)
    tp          = tp1+tp2+tp3+tp4+tp5
    tav         = (ts+tp)/(2*sa**2)
    return tav