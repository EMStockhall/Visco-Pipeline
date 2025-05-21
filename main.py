# MAINCODE 

from Code_Settings import pipeline_settings
import dot11 as dot
from visco_functions import *
import numpy as np
import time

pf = pipeline_settings()
# Get the directory where the current file is located
file_directory = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the file's directory
os.chdir(file_directory)

# ------------------------------------------------------------------------------------------------------------------
aDot = dot.dot()

# Change dot settings
aDot.nMethod = 1*(pf.optimiser_type == "MMFD") + 2*(pf.optimiser_type == "SLP")  \
                                                + 3*(pf.optimiser_type == "SQP") 
aDot.nPrint = 5  # Print only the final objection function value, the contraint final values and the final design variables value

#RPRM
#aDot.nmRPRM[1] = pf.CMIN
aDot.nmRPRM[8] = pf.FDCH
aDot.nmRPRM[9] = pf.FDCHM

#IPRM
aDot.nmIPRM[2] = pf.maxdotIterations
#aDot.nmIPRM[4] = pf.IWRITE
#aDot.nmIPRM[12] = pf.JWRITE

x = pf.startingpoints.copy()   # First starting point in list
xl = pf.lowerBound.copy()   # Lower bound (lower side constraint)
xu = pf.upperBound.copy()   # Upper bound (upper side constraint)

aDot.evaluate = myEvaluate_dot
exit_vals = aDot.dotcall(x, xl, xu, pf.nCons)

# ------------------------------------------------------------------------------------------------------------------
print("EXITING DOT")