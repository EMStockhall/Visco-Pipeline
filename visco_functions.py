# This is the functions file for the Visco-pipeline


from array import array
import csv
from Code_Settings import pipeline_settings
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pathlib
from pyDOE2 import lhs, ccdesign
import sys
sys.path.append("C:\\Program Files\\MSC.Software\\Marc\\2021.4.0\\mentat2021.4\\shlib\\win64")
from py_post import post_open
import scipy.interpolate as sp
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import mean_squared_error
import time
import scipy as sc
#import dot11 as dot
import re

pf = pipeline_settings()



def get_load_data(t_16_file, cbody_name) -> tuple:
    '''
    This function returns the specified contact body displacement with each time increment
    '''
    # Open the t16 file
    t_16_file = post_open(t_16_file)
    t_16_file.moveto(1)

    # Fine the number id of the contact body with the cbody_name
    for j in range(t_16_file.cbodies()):
        if t_16_file.cbody(j).name == cbody_name:
            break

    # Extract the load history from the t16 files
    load_dat = []
    time_dat = []
    for i in range(1, t_16_file.increments()):
        t_16_file.moveto(i)
        load_dat.append(t_16_file.cbody_force(j)[1])
        time_dat.append(t_16_file.time)

    t_16_file.close()

    return np.array(load_dat), np.array(time_dat)

def log_sample(start, end, n) -> array:
    '''
    This function returns a log sampled array of n points between start and end
    '''
    x = np.log10(end+1)
    X = np.linspace(start, x, n)
    X = (10**X) - 1
    return X

def lin_sample(start, end, n) -> array:
    '''
    This function returns a linear sampled array of n points between start and end
    '''
    x = np.linspace(start, end, n)
    return x

def sample_x(start, end, n, strat = "log") -> array:
    '''
    This function samples the data using the specified sampling strategy
    '''
    if strat == "log":
        x_sample = log_sample(start, end, n)
    elif strat == "lin":
        x_sample = lin_sample(start, end, n)
    elif strat == 'linlog':
        x_sample = np.concatenate((lin_sample(start, end, n), log_sample(start, end, n)), axis=None)
    else:
        raise ValueError("Sampling strategy not recognised")
    
    return x_sample

def error(fem, exp, method = "RMSE") -> float:
    '''
    This function calculates the error between the FEM and experimental data
    '''
    if method == "RMSE":
        err = np.sqrt(np.mean((fem - exp)**2))
    elif method == "MAE":
        err = np.mean(np.abs(fem - exp))
    elif method == "MSE":
        err = np.mean((fem - exp)**2)
    elif method == "Weighted":
        err = np.sqrt(np.mean(((fem - exp)/exp)**2))
    else:
        raise ValueError("Error method not recognised")
    
    return err

def modify_prony_series(dat_file, n_terms, prony_terms):
    '''
    This function modifies the prony series in the dat file to match the number of terms specified
    The prony series input is in the form of a list of 2*n_terms values, where the first 2 terms are the g and tau values respectively
    The prony series is in the form of [g1, tau1, g2, tau2, g3, tau3, ...]
    '''
    viscoheader = ['viscelmoon\n',
                   '\n']
    viscofooter = ['mat color\n']
    
    params = []
    for i in range(n_terms):
        params.append(prony_terms[i*2])
        params.append(prony_terms[i*2 + 1])
        params.append(0)
        params.append(0)
    
    viscolines = []
    viscometa = ['         1         %d         %d         0\n' % (n_terms, n_terms)]

    for j in range(n_terms):
        strng = ''.join([' %.15e' % i for i in params[j*4:j*4 + 4]])
        strng = strng.replace('e+0', '+').replace('e-0', '-') + '\n'
        viscolines.append(strng)
    
    wlines = viscoheader + viscometa + viscolines + viscofooter

    f = open(dat_file, 'r')
    lines = f.readlines()
    f.close()

    for i in range(len(lines)):
        if 'viscelmoon' in lines[i]:
            start = i
            continue
        if 'mat color' in lines[i]:
            end = i + 1
            break
    
    lines[start:end] = wlines

    with open(dat_file, 'w') as f:
        for line in lines:
            f.write(line)
        f.close()


def sts_checker() -> bool:
    sts_file = filepaths('fem_sts')
    data = ''
    exit_number = 0
    with open(sts_file,'r') as f:
        data = f.readlines()

    for i in range(len(data)):
        if 'exit number' in data[i]:
            exit_number = int(data[i].replace('Job ends with exit number :',''))

    return exit_number == 3004


def filepaths(*args) -> str:
    output = []

    base_dir = pf.base_directory

    for i in range(len(args)):
        # Pull the sub folder of the fem_file
        if(args[i]) == "fem_folder":
            output.append(base_dir +
                          '/' + pf.fem_folder)
        # Pull the sub folder of the exp_file
        if(args[i]) == "exp_folder":
            output.append(base_dir +
                          '/' + pf.exp_folder)
        # Marc's .t16 output file for the "Numerical" FE model
        # Command to call with function to obtain the filepath specified below
        if (args[i]) == "fem_t16":
            output.append(base_dir +
                          '/' + pf.fem_folder + '/' + pf.nm_t16)
        # The input file for the "Numerical" model
        elif (args[i]) == "fem_dat":
            output.append(base_dir +
                          '/' + pf.fem_folder + '/' + pf.nm_dat)
        # The status file for the "Numerical" model, to obtain the EXIT CODE
        elif (args[i]) == "fem_sts":
            output.append(base_dir +
                          '/' + pf.fem_folder + '/' + pf.nm_sts)
        elif (args[i]) == "exp_file":
            output.append(base_dir +
                          '/' + pf.exp_folder + '/' + pf.expfile)
        elif (args[i]) == "results_folder":
            output.append(base_dir +
                          '/' + pf.results_folder)
        elif(args[i]) == "2021":
            output.append(r"C:\Program Files\MSC.Software\Marc\2021.4.0\mentat2021.4\bin\mentat.bat")  

    if len(output) > 1:
        output = tuple(output)
        return(output)
    else:
        return(output[0])
    
def calcError() -> float:
    '''
    This function calculates the error between the FEM and experimental data
    '''
    fem_file = filepaths('fem_t16')
    exp_file = filepaths('exp_file')
    

    # Get the load history from the FEM simulation
    fem_load, fem_time = get_load_data(fem_file, pf.indName)

    if pf.flipFemForce == True:
        fem_load = -fem_load*pf.forceMul
    else:
        fem_load = fem_load*pf.forceMul
    

    # Get the load history from the experimental data
    exp_data = pd.read_csv(exp_file, sep='\t', header=None).to_numpy()
    exp_time = exp_data[:,0]
    exp_load = exp_data[:,1]


    # Adjust the timing in the fem data to match the experimental data
    if pf.useWholeLoad == True:
        fem_time = fem_time[pf.useWholeLoadStartFem:]
        fem_time = fem_time - fem_time[0]
        fem_load = fem_load[pf.useWholeLoadStartFem:]

    else:
        fem_time = fem_time[pf.maxFemInc:]
        fem_time = fem_time - fem_time[0]
        fem_load = fem_load[pf.maxFemInc:]
        exp_time = exp_time[pf.maxExpForce:]
        exp_time = exp_time - exp_time[0]
        exp_load = exp_load[pf.maxExpForce:]

    # Normalize the data
    if pf.normData == True:
        fem_load = fem_load / np.max(fem_load)
        exp_load = exp_load / np.max(exp_load)
    
    # Interpolate the data to match the time points
    xsample = sample_x(0, pf.sampleTime, pf.samplePoints, strat = pf.sampleStrat)
    fem_interp = sc.interpolate.CubicSpline(fem_time, fem_load)
    exp_interp = sc.interpolate.CubicSpline(exp_time, exp_load)

    # Calculate the error
    err = error(fem_interp(xsample), exp_interp(xsample), method = pf.error_method)

    return err

def log_iteration(params, obj, g):
    save_dir = filepaths('results_folder')
    save_name = 'iterations.txt'

    # Change params back into base 10 from log
    if pf.scalingStrategy == 'log':
        params = np.power(10, params)
    if pf.scalingStrategy == 'linear':
        params = np.multiply(params, pf.weights)

    data = list(params) + [obj] + list(g)

    param_names = [[f'g{i+1}', f'tau{i + 1}'] for i in range(len(params)//2)]
    opt_params = ['OBJ'] + ['G' + str(item) for item in range(len(g))]
    save_header = param_names + opt_params

    save_path = save_dir + '/' + save_name

    if os.path.isfile(save_path) == False:
        f = open(save_path, 'a', newline='')
        write = csv.writer(f, delimiter=',')
        write.writerow(save_header)
        f.close()

    
    f = open(save_path, 'a', newline='')
    write = csv.writer(f, delimiter=',')
    write.writerow(['%.16e' % i for i in data])
    f.close()

    # Make params back into log space
    if pf.scalingStrategy == 'log':
        params = np.log10(params)
    if pf.scalingStrategy == 'linear':
        params = np.divide(params, pf.weights)


def myEvaluate_dot(x, obj, g, param) -> None:

    '''
    This function is used to evaluate the objective function and constraints for the DOT optimisation
    '''
    # Change x back into base 10 from log
    x = np.power(10, x)
    time_before = time.time()

    home_direct = os.getcwd()
    # save_direct = home_direct + '/' + pf.results_folder
    fem_dat, fem_folder = filepaths('fem_dat', 'fem_folder')

    modify_prony_series(fem_dat, pf.pronyTerms, x)

    print("Submitting FEM simulation with parameters: ", x)

    os.chdir(fem_folder)
    os.system("run_marc -j %s -dir %s -sdir %s -nts 8 -nte 8"%(fem_dat, os.getcwd(), os.getcwd()))
    os.chdir(home_direct)

    time.sleep(2)

    # Check if the simulation was successful
    sts_check = sts_checker()
    if sts_check == False:
        print("FEM simulation failed")
        obj.value = 1
        g[0] = 1
    else:
        g[0] = -1
        obj.value = calcError()
        print("FEM simulation completed successfully")
    
    # Make x back into log space
    x = np.log10(x)
    
    print("Begining DOT iteration")
    print("Objective function value: ", obj.value)
        
