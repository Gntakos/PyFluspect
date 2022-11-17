import numpy as np
import pandas as pd
from scipy import io
import os

def read_fixed_input():
    #define spectral regions
    spectral ={
        "wlP": np.arange(400, 2401,1),
        "wlE": np.arange(400, 751,1),
        "wlF": np.arange(640, 851,1)
    }

    #fixed input for SIF simulation with PCA
    pcf = pd.read_csv(os.getcwd() + r'\inputs' + r'\PC_flu.csv')

    #fixed input for FLUSPECT and BSM
    optipar = io.loadmat(os.getcwd() + r'\inputs' + r'\Optipar2021_ProspectPRO_CX.mat')

    #colect
    fixed = {
        "spectral": spectral,
        "pcf": pcf,
        "optipar": optipar,
        "srf_sensors": np.array(['OLCI', 'MSI', 'altum_multispec', 'SLSTR', 'Synergy'])
    }
    return fixed