# MAINCODE 

from Code_Settings import pipeline_settings
import dot11 as dot
from visco_functions import *
import numpy as np
import time
from matplotlib import pyplot as plt

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
aDot.nmIPRM[4] = pf.IWRITE
aDot.nmIPRM[12] = pf.JWRITE

x = pf.startingpoints.copy()   # First starting point in list
xl = pf.lowerBound.copy()   # Lower bound (lower side constraint)
xu = pf.upperBound.copy()   # Upper bound (upper side constraint)

aDot.evaluate = myEvaluate_dot
exit_vals = aDot.dotcall(x, xl, xu, pf.nCons)

# ------------------------------------------------------------------------------------------------------------------
print("EXITING DOT")

# Creating a graph of the final result
final_params = exit_vals[2:]

fem_file = filepaths('fem_t16')
exp_file = filepaths('exp_file')

# Get the load history from the FEM simulation
fem_load, fem_time = get_load_data(fem_file, pf.indName)
# Get the load history from the experimental data
exp_data = pd.read_csv(exp_file, sep=',', header=None).to_numpy()
exp_time = exp_data[:,0]
exp_load = exp_data[:,1]

# Normalize the data
if pf.normData == True:
    fem_load = fem_load / fem_load[pf.maxFemInc]
    exp_load = exp_load / exp_load[pf.maxExpForce]

# Adjust the timing in the fem data to match the experimental data
if pf.useWholeLoad == True:
    fem_time = fem_time[pf.useWholeLoadStartFem:]
    fem_time = fem_time - fem_time[0]
    fem_load = fem_load[pf.useWholeLoadStartFem:]
    exp_time = exp_time[pf.useWholeLoadStartExp:]
    exp_time = exp_time - exp_time[0]
    exp_load = exp_load[pf.useWholeLoadStartExp:]

else:
    fem_time = fem_time[pf.maxFemInc:]
    fem_time = fem_time - fem_time[0]
    fem_load = fem_load[pf.maxFemInc:]
    exp_time = exp_time[pf.maxExpForce:]
    exp_time = exp_time - exp_time[0]
    exp_load = exp_load[pf.maxExpForce:]



# Interpolate the data to match the time points
xsample = sample_x(0, pf.sampleTime, pf.samplePoints, strat = pf.sampleStrat)
fem_interp = sc.interpolate.CubicSpline(fem_time, fem_load)
exp_interp = sc.interpolate.CubicSpline(exp_time, exp_load)

# Change the current working directory to the file's directory
os.chdir(file_directory)

plt.figure(figsize=(10,6))
plt.plot(xsample, fem_interp(xsample), label='FEM Simulation', color='blue')
plt.plot(xsample, exp_interp(xsample), label='Experimental Data', color='orange')
plt.text
plt.xlabel('Time (s)')
plt.ylabel('Load')
plt.title('Load History Comparison')
plt.legend()
plt.grid()
plt.savefig('final_result.pdf')
plt.close()