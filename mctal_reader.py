import numpy as np
from f4enix.output.mctal import Mctal
import matplotlib.pyplot as plt
from pathlib import Path

##################################
# user inputs
##################################

group_structure = 709 # see model for group structures used
tally_number = 134 #134=709 144=725 154=1102 for d-li #204=1102 214=709 224=725 for p-li
current_ua = 10 # current in uA for printing graphs

folder = "bear_postpro/dli_experiment_fzk_jun25"

filename="tuned_dli_1e10_fzk_val_030625_m"

# insert cells in following format: see models for list of cells tallied
cells = [
    {"id":1,  "desc": "copper behind target"},
    {"id":2,  "desc": "lithium target inner"},
    {"id":3,  "desc": "lithium target outer"},
    {"id":4,  "desc": "fe foil"},
    {"id":5,  "desc": "cu foil"},
    {"id":6,  "desc": "rh foil"},
    {"id":7,  "desc": "al foil"},
    {"id":8,  "desc": "dy foil"},
    {"id":9,  "desc": "ni foil"},
    {"id":10, "desc": "au foil"},
    {"id":11, "desc": "y foil"},
    {"id":12, "desc": "sc foil"},
    {"id":13, "desc": "in foil"},
    {"id":14, "desc": "cd foil"},
    {"id":15, "desc": "nb foil"},
    {"id":16, "desc": "mo foil"},
    {"id":17, "desc": "diamond detector sensor"},
    {"id":18, "desc": "mo block"}
] # deuteron model cells
# cells = [
#     {"id": 1,  "desc": "copper behind target"},
#     {"id": 2,  "desc": "lithium target inner"},
#     {"id": 3,  "desc": "lithium target outer"},
#     {"id": 4,  "desc": "fe foil"},
#     {"id": 5,  "desc": "cu foil"},
#     {"id": 6,  "desc": "rh foil"},
#     {"id": 7,  "desc": "al foil"},
#     {"id": 8,  "desc": "dy foil"},
#     {"id": 9,  "desc": "ni foil"},
#     {"id": 10, "desc": "au foil"},
#     {"id": 11, "desc": "y foil"},
#     {"id": 12, "desc": "sc foil"},
#     {"id": 13, "desc": "in foil"},
#     {"id": 14, "desc": "cd foil"},
#     {"id": 15, "desc": "nb foil"},
# ] # proton model cells

##################################
##################################

def get_tally(mctal: Mctal, tally: int) -> tuple:
    # grab the relevant data from the pandas dataframe and make a few lists
    energy = mctal.tallydata.get(tally).get("Energy").tolist()
    value = mctal.tallydata.get(tally).get("Value").tolist()
    relative_error = mctal.tallydata.get(tally).get("Error").tolist()
 
    return energy, value, relative_error

mctal = Mctal(f"{folder}/{filename}")
energy, value, rel_err = get_tally(mctal, tally_number)
proton_flux_scaling = 6.24151e12 # protons/s for 1uA

def get_neutron_flux(cell_numerator):
    n = cell_numerator - 1 
    if group_structure > 500:
        energy_binning = energy[:group_structure]
        flux = np.array(value[(n*group_structure)+n:((n+1)*group_structure)+n])
    else:
        energy_binning = energy[:group_structure+2]
        flux = np.array(value[n*(group_structure+3):(n+1)*(group_structure)+2+(n*3)])
    return energy_binning,flux

def dump_fluxes_file(cell_numerator, output: Path):
    flux = get_neutron_flux(cell_numerator)[1]
    total_flux = sum(flux)

    with open(output, "w") as f:
        count = 1
        # legacy fispact likes reverse order
        for flx in reversed(flux):
            f.write(f"{flx:.5e} ")
            count += 1
            if count > 6:
                f.write("\n")
                count = 1

        f.write(f"\n1.000")
        f.write(f"\nTotal = {total_flux:.5e} [n/cm2/particle]")


def dump_flux(cell_numerator, desc: str, output: Path):
    x, y = get_neutron_flux(cell_numerator)

    with open(output, "w") as f:
        f.write(f"Description: {desc}\n\n")
        f.write(
            f'Note: tallies calculated for per source charged particle. \n'
        )
        f.write(f"  src protons/deuterons @1uA = {proton_flux_scaling:.5e} [particles/s]\n\n")
        f.write(f"Energy [MeV], Flux [n/cm2/particle]\n")
        for energy, flux in zip(x, y):
            f.write(f"{energy:.5e} {flux:.5e}\n")
        f.write(f"Total = {sum(y):.5e} [n/cm2/particle]")


path_fluxes = Path(f"{folder}/fluxes_{group_structure}/")
path_fluxes.mkdir(exist_ok=True)

path_text = Path(f"{folder}/text_{group_structure}/")
path_text.mkdir(exist_ok=True)

path_plots = Path(f"{folder}/plots_{group_structure}_{current_ua}uA/")
path_plots.mkdir(exist_ok=True)

for cell in cells:
    filename = f"{cell['desc'].replace(' ', '_')}.dat"

    output = path_fluxes.joinpath(filename)
    dump_fluxes_file(cell["id"], output)

    output = path_text.joinpath(filename)
    dump_flux(cell["id"],  cell["desc"], output)

    fig, ax1 = plt.subplots(figsize=(8,6))
    current_scaled_flux = [proton_flux_scaling*current_ua*i for i in get_neutron_flux(cell["id"])[1]]
    ax1.step(get_neutron_flux(cell["id"])[0], current_scaled_flux, where='post', label=f'{cell["desc"]}')

    #ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylim(1e4,1e8)
    ax1.set_xlim(0,30)
    ax1.grid()
    ax1.set_ylabel('Flux (n cm$^{-2}$ s$^{-1}$)')
    ax1.set_xlabel('Neutron energy (MeV)')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.04, hspace=0.1)
    plt.savefig(f"{path_plots}/{cell['desc'].replace(' ', '_')}.png")
