import os
import shutil
import numpy as np
import sys
import matplotlib.font_manager
import matplotlib
import matplotlib.pyplot as plt
#plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Helvetica"
plt.rcParams["font.size"] = 22
plt.rcParams["font.weight"] = "normal"
from collections import Counter
from itertools import count
from random import randint
from pathlib import Path
import math

######################
######################
######################
######################

# select materials simulated - material must be present in the input file example directory /materials/{} with stoich/mass/density/timings set up. 
materials = ['au', 'al', 'fe', 'in', 'nb', 'ni', 'dy', 'cd', 'cu'] # all mat ['li','ss','v44','v','au', 'al', 'fe', 'in', 'nb', 'ni', 'rh', 'sc', 'y', 'dy', 'cd','cu'] # other materials: ['ss', 'v44','v','li','lipb','macor']
#materials = ['fe'] 

# select particle from: proton neutron deuteron. Charged particles still in development 
projectile = 'neutron' 

# comment for input file
input_comment = 'proton unfolded simulation'
# select template - 'approx' for single scoping timescales, 'pli_experiment' for march 2024 irradiation, 'dli_experiment' for nov24 irradiation
input_file_template = 'pli_experiment' 
# if using the 'example' template, set the irradiation time in an array of one timing (mins) and one proton/deuteron current (uA)
irradiation_time_mins = 120
current_in_ua = 20

# folder path and filenames for the flux to be used - make sure group structures line up
#flux_folder_path = '/Users/ljb841@student.bham.ac.uk/fispact/WORKSHOP/fluxes/experiment_shielded_2024/fluxes_1102'
#flux_folder_path = '/Users/ljb841@student.bham.ac.uk/mcnp/workshop/uBB/bear_postpro/dli_experiment_jan25/1e10/fluxes_1102'
flux_folder_path = '/Users/ljb841@student.bham.ac.uk/spectra-uf-master/examples/pli_mar24/results_175gs_250425'

# set up to use 'tendl21' (1102-group) or 'endfb8' (709-group) or 'irdff2' (725-group) libraries
library = 'irdff2' 
group = 725

flux_benchmark_factor =  0.0309089 # 0.0309089 for protons # 0.119855(85mb) or 0.169795(60mb) for deuterons 

######################
######################
######################
######################

if input_file_template == 'pli_experiment':
    current_ua = [5.5,0,9,0,10]
    irrad_time = [20,36.3,67,10.3,41.5]
else:
    current_ua = [current_in_ua]
    irrad_time = [irradiation_time_mins]

def fispact_setup(material):

    # copy example input file, and mark paths for input/fluxes/arb_flux/files files
    input_file_path = shutil.copyfile(f'/Users/ljb841@student.bham.ac.uk/fispact/WORKSHOP/uBB/materials/{input_file_template}/uBB_{material}.i', f'{material}_{input_file_template}.i')
    files_file_path = '/Users/ljb841@student.bham.ac.uk/fispact/WORKSHOP/uBB/files'
    arb_flux_file_path = f'/Users/ljb841@student.bham.ac.uk/fispact/WORKSHOP/uBB/arb_flux_{group}'
    #fluxes_file_path = f'{flux_folder_path}/{material}_foil.dat'
    fluxes_file_path = f'{flux_folder_path}/unfolded_{group}_GRAVEL.txt'

    # calculate flux values from the fluxes file, to multiply the fispact input file fluxes by
    with open(fluxes_file_path, 'r') as filename3:
        fluxes_file = filename3.readlines()
    if group == 725:
        initial_flux_value = float(fluxes_file[122].split()[2])
    if group == 709:
        initial_flux_value = float(fluxes_file[120].split()[2])
    if group == 1102:
        initial_flux_value = float(fluxes_file[185].split()[2])
    if input_file_template == 'dli_experiment':
        adjusted_total_flux_per_src = initial_flux_value*flux_benchmark_factor
    else:
        fluence_factor = flux_benchmark_factor*6.24151e+12
        final_flux_values = []
        for i in current_ua:
            final_flux_values.append(initial_flux_value*(i)*fluence_factor)

    # small switch to adjust the arb_flux file so will work for charged particles - NEED DEV
    if projectile != 'neutron':
        with open(arb_flux_file_path,'r') as arb_flux_file_raw:
            arb_flux_file = arb_flux_file_raw.readlines()
            arb_flux_file[group+1:] = fluxes_file
        with open(arb_flux_file_path,'w') as arb_flux_file_raw:
            arb_flux_file_raw.writelines(arb_flux_file)

    # make changes to the input file copied from the example input file directory
    with open(input_file_path,'r') as input_file_raw:
        input_file = input_file_raw.readlines()
    flux_keyword_line = 25+int(input_file[17-1].split()[2])
    # general changes non-specific to the experiment and example input file
    input_file[10-1] = '<< ALLDISPEN 40 >> \n'
    input_file[13-1] = f'* {input_comment} \n'
    #if j == 'li':
    input_file[14-1] = '<< NUCGRAPH 1 0.01 1 1 >> \n'
    input_file[flux_keyword_line-6] = 'DOSE 2 0.3 \n'
    input_file[flux_keyword_line-5] = 'UNCERTAINTY 2 \n'
    if projectile == 'neutron':
        input_file[5-1] = 'PROJ 1 \n'
        input_file[6-1] = '<< GRPCONVERT 1102 162 >>\n'
        input_file[8-1] = f'GETXS 1 {group} \n'
    else:
        input_file[5-1] = 'PROJ 3 \n'
        input_file[6-1] = 'GRPCONVERT 1102 162 \n'
        input_file[8-1] = 'GETXS 1 162 \n' 
    # changes specific to the experiment being modelled
    if input_file_template == 'approx':
        input_file[flux_keyword_line-1] = f'FLUX {final_flux_values[0]} \n'
        input_file[flux_keyword_line+1] = f'TIME {irrad_time[0]} MINS ATOMS \n'
    if input_file_template == 'pli_experiment':
        input_file[flux_keyword_line-1]  = f'FLUX {final_flux_values[0]} \n'
        input_file[flux_keyword_line+1]  = f'TIME {irrad_time[0]} MINS ATOMS \n'
        input_file[flux_keyword_line+2]  = f'FLUX {final_flux_values[1]} \n'
        input_file[flux_keyword_line+4]  = f'TIME {irrad_time[1]} MINS ATOMS \n'
        input_file[flux_keyword_line+5]  = f'FLUX {final_flux_values[2]} \n'
        input_file[flux_keyword_line+7]  = f'TIME {irrad_time[2]} MINS ATOMS \n'
        input_file[flux_keyword_line+8]  = f'FLUX {final_flux_values[3]} \n'
        input_file[flux_keyword_line+10] = f'TIME {irrad_time[3]} MINS ATOMS \n'
        input_file[flux_keyword_line+11] = f'FLUX {final_flux_values[4]} \n'
        input_file[flux_keyword_line+13] = f'TIME {irrad_time[4]} MINS ATOMS \n'
    if input_file_template == 'dli_experiment':
        for n in range(5530):
            charged_src_particles = float(input_file[flux_keyword_line+(2*n)-1].split()[1])
            neutron_flux = charged_src_particles*adjusted_total_flux_per_src
            input_file[flux_keyword_line+(2*n)-1] = f"FLUX {neutron_flux} ATOMS \n"
    with open(input_file_path, 'w') as input_file_raw:
        input_file_raw.writelines(input_file)

    # make changes to the files file for the duration of the simulations
    with open(files_file_path,'r') as files_file_raw:
        files_file = files_file_raw.readlines()
    # switches on the files file for neutrons
    if projectile == 'neutron':
        files_file[11-1] = f'fluxes {fluxes_file_path} \n'
        files_file[12-1] = '# arb_flux arb_flux \n'
        files_file[8-1] = 'prob_tab  ../../nuclear_data/tendl21data/tp-1102-294 \n'
        files_file[18-1] = '# enbins ../../nuclear_data/IRDFF-II/ebins_725 \n' 
        files_file[15-1] = 'dk_endf ../../nuclear_data/decay2020/decay_2020 \n'
        files_file[2-1] = 'ind_nuc  ../../nuclear_data/decay2020/decay_2020_index.txt \n'
        files_file[5-1] = 'xs_endf ../../nuclear_data/tendl21data/gendf-1102 \n' 
        if library == 'irdff2':
            files_file[5-1] = 'xs_endf ../../nuclear_data/IRDFF-II/irdff-II_725-n \n'
            files_file[18-1] = 'enbins ../../nuclear_data/IRDFF-II/ebins_725 \n' 
            files_file[15-1] = 'dk_endf ../../nuclear_data/IRDFF-II/decay_irdff-II \n'
            files_file[2-1] = 'ind_nuc  ../../nuclear_data/IRDFF-II/IRDFF_index \n'
        if library == 'endfb8':
            files_file[5-1] = 'xs_endf ../../nuclear_data/ENDFB80data/endfb80-n/gxs-709 \n' 
        if library == 'tend21':
            files_file[5-1] = 'xs_endf ../../nuclear_data/tendl21data/gendf-1102 \n' 
    # switches on the files file for charged particles - NEED DEV 
    else:
        files_file[5-1] = 'xs_endf ../../nuclear_data/p-tendl2019/gxs-162 \n'
        files_file[8-1] = 'prob_tab  ../../nuclear_data/tendl19data/tp-1102-294 \n'
        files_file[11-1] = f'fluxes {fluxes_file_path} \n'
        files_file[12-1] = 'arb_flux arb_flux \n'
    with open(files_file_path, 'w') as files_file_raw:
        files_file_raw.writelines(files_file)

# Runs FISPACT for all requested materials
for j in materials:
    print(f'Running FISPACT for {j}_{input_file_template}.i')
    fispact_setup(j)
    os.system(f'$fispact {j}_{input_file_template}.i &')