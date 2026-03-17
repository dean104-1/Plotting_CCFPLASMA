#On carpenter do: module load cray-python
import numpy as np
import matplotlib
matplotlib.use('Agg')  #display not required
import matplotlib.pyplot as plt
from multiprocessing import Pool
import util_functions

slices = []
taps = []

timestep = 1e-7
num_slices = 500
num_taps = 10000
filename_slices = "../5000Hz_276mm/outputs/outputs_060000/slice_K151_060000"
filename_taps = "../5000Hz_276mm/outputs/outputs_060000/taps_K151_060000"

slices = util_functions.loadslices(filename_slices,num_slices)
taps = util_functions.loadslices(filename_taps,num_taps)

print("Done loading data")

def plot_dens_grad(nn):
    slice_data = slices[nn]

    v = np.linspace(0, 2.0, 300, endpoint=True)
    radius = np.sqrt(slice_data["Y"][:,0,:]**2 + slice_data["Z"][:,0,:]**2)

    rhoGrad = util_functions.computeSchlieren(slice_data["NJ"], slice_data["NL"], slice_data["X"][:,0,:], radius, slice_data["Q"][:,0,:,0])

    plt.rcParams.update({'font.size': 16,'axes.labelsize': 16,
                         'axes.titlesize': 16,'xtick.labelsize': 16,
                         'ytick.labelsize': 16})

    fig, ax = plt.subplots(figsize=(20, 3.2), constrained_layout=True)
    ax.contourf(slice_data["X"][:,0,:], radius, rhoGrad[:,0,:],v, cmap='gray', extend='both')
    plt.xlim(350, 600)
    plt.ylim(30, 70)
    plt.title(r"$\nabla \rho /\rho_{\infty}$")
    plt.xlabel('x [mm]')
    plt.ylabel('radius [mm]')
    plt.savefig(f'images/myplot_{nn:02d}.png', dpi=300)
    plt.close()

def plot_tap_history(nn):
    print("begin tap history function")
    slice_data = slices[nn]

    rhoinf = 0.03754 #kg/m^3
    Tinf = 51        #K
    ainf = 143.150   #m/s
    R = 287.0        #J/kg-K
    Cv = 717         #J/kg-K
    Pinf = rhoinf*R*Tinf

    tapnum = [700,900]

    P = []
    
    print("beginning tap processing")
    for kk in range(len(tapnum)):

        rho_star = []
        rho_star_u_star = []
        rho_star_v_star = []
        rho_star_w_star = []
        rho_star_e_star = []

        for jj in range(num_taps):

            rho_star.append(taps[jj]["Q"][tapnum[kk],0,0,0])
            rho_star_u_star.append(taps[jj]["Q"][tapnum[kk],0,0,1])
            rho_star_v_star.append(taps[jj]["Q"][tapnum[kk],0,0,2])
            rho_star_w_star.append(taps[jj]["Q"][tapnum[kk],0,0,3])
            rho_star_e_star.append(taps[jj]["Q"][tapnum[kk],0,0,4])


        rho = np.array(rho_star)*rhoinf
        u_star = np.array(rho_star_u_star)/np.array(rho_star)
        u = u_star*ainf
        v_star = np.array(rho_star_v_star)/np.array(rho_star)
        v = v_star*ainf
        w_star = np.array(rho_star_w_star)/np.array(rho_star)
        w = w_star*ainf
        e_star = np.array(rho_star_e_star)/np.array(rho_star)
        e = e_star*ainf**2
        temp = (e - 0.5*(u**2 + v**2 + w**2))/Cv

        P.append(rho*R*temp)
        print("processed one tap")

    print("begin plotting")

    v = np.linspace(0, 2, 300, endpoint=True)
    radius = np.sqrt(slice_data["Y"][:,0,:]**2 + slice_data["Z"][:,0,:]**2)

    rhoGrad = util_functions.computeSchlieren(slice_data["NJ"], slice_data["NL"], slice_data["X"][:,0,:], radius, slice_data["Q"][:,0,:,0])

    timevec = np.arange(1, 10001) * timestep

    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16
    })

    fig, axes = plt.subplots(3, 1, figsize=(8, 6))

    axes[0].contourf(slice_data["X"][:,0,:], radius, rhoGrad[:,0,:],v, cmap='gray', extend='both')
    #axes[0].scatter([380.0],[33.34])
    axes[0].scatter(taps[0]["X"][tapnum[0]].flatten(), np.sqrt(taps[0]["Y"][tapnum[0]].flatten()**2 + taps[0]["Z"][tapnum[0]].flatten()**2), color='magenta')
    axes[0].scatter(taps[0]["X"][tapnum[1]].flatten(), np.sqrt(taps[0]["Y"][tapnum[1]].flatten()**2 + taps[0]["Z"][tapnum[1]].flatten()**2), color='green')
    axes[0].set_title(r"$\nabla \rho /\rho_{\infty}$")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("r [mm]")
    axes[0].set_xlim(390, 590)

    axes[1].plot(timevec, P[0]/Pinf, color='magenta')
    axes[1].axvline(x= timestep*20*nn, linestyle='--', linewidth=1, label='Vertical line')
    axes[1].set_ylabel("$P/P_{\infty}$")
    axes[1].set_xlim(0,0.001)
    axes[1].grid(True)

    axes[2].plot(timevec, P[1]/Pinf, color='green')
    axes[2].axvline(x= timestep*20*nn, linestyle='--', linewidth=1, label='Vertical line')
    axes[2].set_ylabel("$P/P_{\infty}$")
    axes[2].set_xlabel("time [s]")
    axes[2].set_xlim(0,0.001)
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f'images/myplot_{nn:02d}.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    num_workers = 128

    with Pool(processes=num_workers) as pool:
        pool.map(plot_tap_history, range(num_slices))
        #pool.map(plot_dens_grad, range(num_slices))

    print("All slices processed in parallel.")
