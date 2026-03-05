# Preprocessing / parameter sweep to reproduce Figure 3A (Deco et al., 2018)
#
# This script scans global coupling we (G in the paper) for the Placebo condition,
# using two fitting targets:
#   - FC similarity (Pearson similarity, higher is better)
#   - swFCD distance (KS statistic, lower is better)
#
# The goal is to identify a reasonable we to use in later stages (e.g., Fig 3B/3E).

from my_setup import *

# --------------------------------------------------------------------------
# Configuration (tune these to reduce runtime during debugging)
# --------------------------------------------------------------------------
WE_START = 0.0
WE_END = 2.4
WE_STEP = 0.3        # original Matlab sweep is much finer (e.g., 0.025)

# Number of simulated subjects (trials) per parameter value.
# Keep it small (e.g., 1–2) for debugging; set to NumSubjects for full reproduction.
NUM_SIM_SUBJECTS = None  # None -> use NumSubjects

# --------------------------------------------------------------------------
# Main routine
# --------------------------------------------------------------------------
def prepro_G_Optim():
    # Cache pattern for J balancing (FIC) per we
    J_fileNames = str(out_dir / "J_Balance_we{}.mat")

    # Observables to compute and whether to apply band-pass filtering
    distanceSettings = {
        'FC': (FC, False),
        'swFCD': (swFCD, True),
    }

    # Parameter grid
    WEs = np.arange(WE_START, WE_END + WE_STEP, WE_STEP)

    # FIC balancing for each we: returns dict {we: {'we': we, 'J': ...}}
    BalanceFIC.verbose = True
    balancedParms = BalanceFIC.Balance_AllJ9(C, WEs, baseName=J_fileNames)

    # Make modelParms list aligned with WEs order (avoid relying on dict iteration order)
    modelParms = [balancedParms[float(we)] for we in WEs]

    # Decide how many simulated subjects to run per we
    NumSimSubjects = NumSubjects if NUM_SIM_SUBJECTS is None else int(NUM_SIM_SUBJECTS)
    NumSimSubjects = max(1, min(NumSimSubjects, NumSubjects))

    print("\n\n####################################")
    print("# Compute G_Optim (prepro_G_Optim)")
    print("####################################\n")

    fitting = optim1D.distanceForAll_Parms(
        tc_transf_PLA,
        WEs,
        modelParms,
        NumSimSubjects=NumSimSubjects,
        observablesToUse=distanceSettings,
        parmLabel='we',
        outFilePath=str(out_dir),
        fileNameSuffix=''
    )

    optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    print("Optimal:\n", optimal)

    filePath = str(out_dir / "DecoEtAl2018_fneuro.mat")
    sio.savemat(
        filePath,
        {
            'we': WEs,
            'fitting_PLA': fitting['FC'],
            'FCDfitt_PLA': fitting['swFCD'],
        }
    )
    print(f"DONE!!! (file: {filePath})")

if __name__ == '__main__':
    prepro_G_Optim()
