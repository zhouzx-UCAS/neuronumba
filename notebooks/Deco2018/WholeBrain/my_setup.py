# Setup for the code from the paper
#
# [DecoEtAl_2018] Whole-brain multimodal neuroimaging model using serotonin receptor maps explains non-linear functional effects of LSD (2018) Current Biology
#
# Translated to Python & refactoring by Gustavo Patow
# Cleaned up for notebook use: keep original names, improve readability,
# and allow selecting a smaller fMRI subset file.

import numpy as np
import hdf5storage as sio
from numba import jit
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
inFilePath = '.'
outFilePath = 'out4'
out_dir = Path(outFilePath)
out_dir.mkdir(parents=True, exist_ok=True)

# Choose which fMRI file to load:
#   - 'LSDnew.mat'       : original dataset (15 subjects x 6 conditions)
#   - 'LSDnew_sub1.mat'  : 1 subject x 6 conditions
#   - 'LSDnew_sub2.mat'  : 2 subjects x 6 conditions
fmriFileName = 'LSDnew_sub1.mat'

# Condition indices (0-based indexing in Python)
# Original Matlab code used [2, 5] (1-based), hence [1, 4] here.
PLACEBO_cond = 4
LSD_cond = 1

# --------------------------------------------------------------------------
# WholeBrain modules wiring (model -> scheme -> integrator -> simulateBOLD/optim/FIC)
# --------------------------------------------------------------------------
# import WholeBrain.Models.DynamicMeanField as DMF
import WholeBrain.Models.serotonin2A as serotonin2A
# import WholeBrain.Models.Couplings as Couplings

import WholeBrain.Integrators.EulerMaruyama as scheme
scheme.neuronalModel = serotonin2A

import WholeBrain.Integrators.Integrator as integrator
integrator.integrationScheme = scheme
integrator.neuronalModel = serotonin2A
integrator.verbose = False

import WholeBrain.Utils.BOLD.BOLDHemModel_Stephan2007 as Stephan2007
import WholeBrain.Utils.simulate_SimAndBOLD as simulateBOLD
simulateBOLD.integrator = integrator
simulateBOLD.BOLDModel = Stephan2007

import WholeBrain.Optimizers.ParmSweep as optim1D
optim1D.simulateBOLD = simulateBOLD
optim1D.integrator = integrator

import WholeBrain.Utils.FIC.BalanceFIC as BalanceFIC
BalanceFIC.integrator = integrator
import WholeBrain.Utils.FIC.Balance_DecoEtAl2014 as Deco2014Mechanism
BalanceFIC.balancingMechanism = Deco2014Mechanism

# --------------------------------------------------------------------------
# Filters and Observables
# --------------------------------------------------------------------------
import WholeBrain.Observables.BOLDFilters as filters
filters.k = 2          # 2nd order Butterworth
filters.flp = 0.01     # band-pass low cutoff (Hz)
filters.fhi = 0.1      # band-pass high cutoff (Hz)
filters.TR = 2.0       # sampling interval (s)

import WholeBrain.Observables.FC as FC
import WholeBrain.Observables.swFCD as swFCD

# --------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------
@jit(nopython=True)
def initRandom():
    np.random.seed(3)  # original code used 13 in some versions

def LR_version_symm(TC):
    """
    Build a left-right symmetric ordering for an AAL signal matrix.

    Parameters
    ----------
    TC : array, shape (90, T)
        Time series for 90 AAL regions.

    Returns
    -------
    symLR : array, shape (90, T)
        Reordered time series: first 45 and last 45 arranged in a symmetric LR scheme.
    """
    odd = np.arange(0, 90, 2)
    even = np.arange(1, 90, 2)[::-1]
    symLR = np.zeros((90, TC.shape[1]))
    symLR[0:45, :] = TC[odd, :]
    symLR[45:90, :] = TC[even, :]
    return symLR

def transformEmpiricalSubjects(tc_aal, cond, NumSubjects):
    """
    Convert MATLAB cell array tc_aal into a Python dict: subject -> (90, T) array,
    applying LR_version_symm() to each subject.

    Parameters
    ----------
    tc_aal : object array, shape (NumSubjects, NumConditions)
        Each entry is a (90, T) ndarray.
    cond : int
        Condition column index (0-based).
    NumSubjects : int
        Number of subjects to include.

    Returns
    -------
    transformed : dict[int, ndarray]
        Keys are subject indices 0..NumSubjects-1.
    """
    transformed = {}
    for s in range(NumSubjects):
        transformed[s] = LR_version_symm(tc_aal[s, cond])
    return transformed

# --------------------------------------------------------------------------
# Initialization: load data and prepare transformed empirical signals
# --------------------------------------------------------------------------
initRandom()

# Structural connectivity
print(f"Loading {inFilePath}/all_SC_FC_TC_76_90_116.mat")
sc90 = sio.loadmat(inFilePath + '/all_SC_FC_TC_76_90_116.mat')['sc90']
C = sc90 / np.max(sc90[:]) * 0.2
serotonin2A.setParms({'SC': C})
serotonin2A.couplingOp.setParms(C)

# Receptor map
print(f"Loading {inFilePath}/mean5HT2A_bindingaal.mat")
mean5HT2A_aalsymm = sio.loadmat(inFilePath + '/mean5HT2A_bindingaal.mat')['mean5HT2A_aalsymm']
serotonin2A.Receptor = (mean5HT2A_aalsymm[:, 0] / np.max(mean5HT2A_aalsymm[:, 0])).flatten()

# fMRI data (tc_aal)
print(f"Loading {inFilePath}/{fmriFileName}")
LSDnew = sio.loadmat(inFilePath + '/' + fmriFileName)
tc_aal = LSDnew['tc_aal']

(N, Tmax) = tc_aal[0, 0].shape
print(f"tc_aal is {tc_aal.shape} and each entry has N={N} regions and Tmax={Tmax}")

NumSubjects = tc_aal.shape[0]
print(f"Simulating {NumSubjects} subjects!")

# Default: placebo parameters
serotonin2A.setParms({'S_E': 0., 'S_I': 0.})

# Precompute transformed empirical signals for placebo and LSD conditions
tc_transf_PLA = transformEmpiricalSubjects(tc_aal, PLACEBO_cond, NumSubjects)
tc_transf_LSD = transformEmpiricalSubjects(tc_aal, LSD_cond, NumSubjects)
