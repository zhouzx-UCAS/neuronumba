# Gain fitting for Figure 3B (Deco et al., 2018)
#
# Fix global coupling we=2.1 (G in the paper), balance FIC (J) once,
# then sweep excitatory gain modulation S_E (wge in original code),
# fitting swFCD (KS distance) for Placebo and LSD conditions.

from my_setup import *

# --------------------------------------------------------------------------
# Configuration (edit here during debugging)
# --------------------------------------------------------------------------
WE_FIXED = 2.1

S_E_START = 0.0
S_E_STEP = 0.02
S_E_END = 0.4

# Number of simulated subjects per parameter value.
NUM_SIM_SUBJECTS = None  # None -> use NumSubjects

# --------------------------------------------------------------------------
# Core function
# --------------------------------------------------------------------------
def fitting_ModelParms(tc_transf, suffix):
    # Balance FIC J at fixed we (cached by @loadOrCompute; file name is last positional arg)
    we_label = str(np.round(WE_FIXED, 2))
    J_fileName = str(out_dir / f"J_Balance_we{we_label}.mat")

    balancedG = BalanceFIC.Balance_J9(WE_FIXED, C.shape[0], False, J_fileName)
    serotonin2A.setParms(balancedG)

    # Baseline (placebo) receptor modulation
    serotonin2A.setParms({'S_E': 0., 'S_I': 0.})

    # Use swFCD only (with filtering)
    distanceSettings = {'swFCD': (swFCD, True)}

    # Sweep S_E
    S_Es = np.arange(S_E_START, S_E_END + S_E_STEP, S_E_STEP)
    serotoninParms = [{'S_I': 0., 'S_E': S_E} for S_E in S_Es]

    NumSim = NumSubjects if NUM_SIM_SUBJECTS is None else int(NUM_SIM_SUBJECTS)
    NumSim = max(1, min(NumSim, NumSubjects))

    fitting = optim1D.distanceForAll_Parms(
        tc_transf,
        S_Es,
        serotoninParms,
        NumSimSubjects=NumSim,
        observablesToUse=distanceSettings,
        parmLabel='S_E',
        outFilePath=str(out_dir),
        fileNameSuffix=suffix
    )

    filePath = str(out_dir / f"DecoEtAl2018_fitting{suffix}.mat")
    sio.savemat(
        filePath,
        {
            'S_E': S_Es,
            'fitting_FCD': fitting['swFCD'],
        }
    )
    print(f"DONE!!! (file: {filePath})")

if __name__ == '__main__':
    fitting_ModelParms(tc_transf_PLA, '_PLA')
    fitting_ModelParms(tc_transf_LSD, '_LSD')
