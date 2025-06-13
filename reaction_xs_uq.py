import sandy
import os
os.environ['NJOY'] = '/Users/ljb841@student.bham.ac.uk/NJOY2016/bin/njoy'
import matplotlib.pyplot as plt
matplotlib.rc("font", family="sans-serif",weight='normal',size=20)
plt.rcParams["font.sans-serif"] = "Helvetica"
import seaborn as sns
import numpy as np
import csv

# user inputs
ek=sandy.energy_grids.VITAMINJ175
library = 'irdff_2' #irdff_2 endfb_71 endfb_80 jendl_40u jeff_33 tendl_21
#material_list = [491150, 
#                 661640,
#                 791970,
#                 491150,
#                 290650,
#                 260560,
#                 130270,
#                 791970,
#                 410930,
#                 280580] 
#mt_values_list = [[102],
#                  [102],
#                  [102],
#                  [4],
#                  [103],
#                  [103],
#                  [107],
#                  [16],
#                  [16],
#                  [16]]

material_list = [260560]
mt_values_list = [[103]]
#material_list = [491150]
#mt_values_list = [[4,102]]

# get covariance and standard deviation data from a material endf6 file
def get_cov_data(material,mt_values):
    mts = mt_values
    try:
        endf_file = sandy.get_endf6_file(library, "xs", material)
        print(endf_file)
        ekws = dict(ek=ek)
        err = endf_file.get_errorr(temperature=0,err=1,chi=False, nubar=False, prod=False,mubar=False, errorr_kws=ekws,verbose=False)["errorr33"]
    except:
        return [],np.array([])
    covariance = err.get_cov()
    std = covariance.get_std().reset_index().query("MT in @mts")
    std["MT"] = std["MT"].astype("category")
    std["STD"] *= 100
    stdev_array = np.array(std["STD"])
    print(f'-----> found reactions for material {material}: {std["MT"].cat.categories}')
    return covariance,stdev_array

# plot covariance matrix from a material endf file
def plot_cov_matrix(material,mt_values):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    cov =  get_cov_data(material,mt_values)[0]
    mask = cov.data.index.get_level_values("MT") != 1
    ng = cov.data.index.get_level_values("MT")[~mask].size
    nr = cov.data.index.get_level_values("MT").unique().size - 1
    data = cov.get_corr().data.iloc[mask, mask]
    sns.heatmap(data=data, vmin=-1, vmax=1, cmap="bwr", ax=ax)
    for i in range(1, nr):
        ax.axvline(ng * i, color="k", ls="--", lw=.5)
        ax.axhline(ng * i, color="k", ls="--", lw=.5)
    fig.tight_layout()
    fig.savefig('cov_matrix.png')

# extract stdev data from covariances and return the stdev array split by MT reactions
def extract_stdev_data(material,mt_values):
    stdev_array = get_cov_data(material,mt_values)[1]
    if stdev_array.size > 0:
        number_of_arrays = len(stdev_array)/(len(ek)-1)
        if number_of_arrays > 1:
            #stdev_array_split = np.array_split(stdev_array, len(stdev_array)/(len(ek)-1))
            stdev_array_split = np.split(stdev_array, number_of_arrays)
        if number_of_arrays == 1:
            stdev_array_split = [stdev_array]
        #stdev_array_transposed = stdev_array.ravel()[None]
        return stdev_array_split
    else:
        print(f'-----> reactions not found for material {material}')
        return None

# export uncert data to one csv and plot uncertainty percentages
def export_and_plot(material_list,mt_values_list):
    open('uncertainty.csv','w').close()
    np.savetxt("uncertainty_group_structure.csv", ek.ravel()[None],delimiter=',')
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

    # loop through specified materials and MT values
    for material,material_mt_values in zip(material_list,mt_values_list):
        split_uncertainty_arrays = extract_stdev_data(material,material_mt_values)
        if split_uncertainty_arrays != None:

            # plot uncertainty data
            for mt_iterator in range(len(split_uncertainty_arrays)):
                ax.stairs(split_uncertainty_arrays[mt_iterator], ek,label=f'{material} MT:{material_mt_values[mt_iterator]}')

            # export data to one csv file
            for xs_stdev in split_uncertainty_arrays:
                print(ek[171],xs_stdev[171])
                with open('uncertainty.csv','a',newline='') as f:
                    writer=csv.writer(f,delimiter=',' )
                    writer.writerow(xs_stdev*(1/100))
        else:
            continue
            #print(f'-----> reactions not found for material {material}')
    ax.legend()
    ax.set(xlim=(1e-1, 2e7), xscale="log", ylim=[0, 100], ylabel="standard deviation / $\%$", xlabel="energy / $eV$")
    fig.tight_layout()
    fig.savefig('percentage_uncert.png')

export_and_plot(material_list,mt_values_list)
