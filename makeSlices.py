import config as cfg
import numpy as np
import matplotlib
matplotlib.use('Agg')  #display not required
import matplotlib.pyplot as plt
from multiprocessing import Pool
import util_functions
import h5py

filename_taps = f"{cfg.basename}/outputs_0090000/taps_K151_0090000"
filename_slices = f"{cfg.basename}/outputs_0090000/slices_K151_0090000"

loadpath_taps = util_functions.loadslices_h5(filename_taps,30000)
with h5py.File(loadpath_taps, 'r') as hf:
    NJ_taps = hf.attrs['NJ']
    NK_taps = hf.attrs['NK']
    NL_taps = hf.attrs['NL']
    XLOC_taps = hf['X'][:,:,:]
    YLOC_taps = hf['Y'][:,:,:]
    ZLOC_taps = hf['Z'][:,:,:]
    P_taps = hf['p'][:,cfg.tapnum_vec,0,0]
    
loadpath_slices = util_functions.loadslices_h5(filename_slices,1500)
with h5py.File(loadpath_slices, 'r') as hf:
    NJ_slices = hf.attrs['NJ']
    NK_slices = hf.attrs['NK']
    NL_slices = hf.attrs['NL']
    XLOC_slices = hf['X'][:,:,:]
    YLOC_slices = hf['Y'][:,:,:]
    ZLOC_slices = hf['Z'][:,:,:]
    rho_slices = hf['rho'][:,:,:,:]
        
print('--Done Loading Data--')


def plot_dens_grad(nn):
    slice_data = rho_slices[nn,:,:,:]

    v = np.linspace(0, 2.0, 300, endpoint=True)
    radius = np.sqrt(YLOC_slices**2 + ZLOC_slices**2)

    rhoGrad = util_functions.computeSchlieren(NJ_slices, NL_slices, XLOC_slices[:,0,:], radius[:,0,:], slice_data[:,0,:])

    plt.rcParams.update({'font.size': 16,'axes.labelsize': 16,
                         'axes.titlesize': 16,'xtick.labelsize': 16,
                         'ytick.labelsize': 16})

    fig, ax = plt.subplots(figsize=(20, 3.81), constrained_layout=True)
    ax.contourf(XLOC_slices[:,0,:], radius[:,0,:], rhoGrad[:,0,:]/cfg.rhoinf, v, cmap='gist_rainbow', extend='both')
    plt.xlim(380, 590)
    plt.ylim(30, 70)
    plt.title(r"$\nabla \rho /\rho_{\infty}$")
    plt.xlabel('x [mm]')
    plt.ylabel('radius [mm]')
    plt.savefig(f'images/myplot_{nn:02d}.png', dpi=300)
    plt.close()
     
def plot_tap_history(nn):
    slice_data = rho_slices[nn,:,:,:]

    v = np.linspace(0, 2, 300, endpoint=True)
    radius = np.sqrt(YLOC_slices**2 + ZLOC_slices**2)
    radius_tap = np.zeros(2)
    radius_tap[0] = np.sqrt(YLOC_taps[cfg.tapnum_vec[0],:,:]**2 + ZLOC_taps[cfg.tapnum_vec[0],:,:]**2).item()
    radius_tap[1] = np.sqrt(YLOC_taps[cfg.tapnum_vec[1],:,:]**2 + ZLOC_taps[cfg.tapnum_vec[1],:,:]**2).item()

    rhoGrad = util_functions.computeSchlieren(NJ_slices, NL_slices, XLOC_slices[:,0,:], radius[:,0,:], slice_data[:,0,:])

    timevec = np.arange(1, 20001) * cfg.timestep

    plt.rcParams.update({'font.size': 16,'axes.labelsize': 16,
                         'axes.titlesize': 16,'xtick.labelsize': 16,
                         'ytick.labelsize': 16})

    fig, axes = plt.subplots(3, 1, figsize=(8, 6))

    axes[0].contourf(XLOC_slices[:,0,:], radius[:,0,:], rhoGrad[:,0,:]/cfg.rhoinf,v, cmap='gray', extend='both')
    axes[0].scatter(XLOC_taps[cfg.tapnum_vec[0],:,:].flatten(), radius_tap[0] ,color='magenta')
    axes[0].scatter(XLOC_taps[cfg.tapnum_vec[1],:,:].flatten(), radius_tap[1] ,color='green')
    axes[0].set_title(r"$\nabla \rho /\rho_{\infty}$")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("r [mm]")
    axes[0].set_xlim(390, 590)

    axes[1].plot(timevec, P_taps[0:20000,0]/cfg.Pinf, color='magenta')
    axes[1].axvline(x= cfg.timestep*20*nn, linestyle='--', linewidth=1, label='Vertical line')
    axes[1].set_ylabel("$P/P_{\infty}$")
    axes[1].set_xlim(0,0.002)
    axes[1].set_xticks([0.0,0.001,0.002])
    axes[1].grid(True)

    axes[2].plot(timevec, P_taps[0:20000,0]/cfg.Pinf, color='green')
    axes[2].axvline(x= cfg.timestep*20*nn, linestyle='--', linewidth=1, label='Vertical line')
    axes[2].set_ylabel("$P/P_{\infty}$")
    axes[2].set_xlabel("time [s]")
    axes[2].set_xlim(0,0.002)
    axes[2].set_xticks([0.0,0.001,0.002])
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f'images/myplot_{nn:02d}.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    num_workers = 64

    with Pool(processes=num_workers) as pool:
        #pool.map(plot_tap_history, range(10))
        pool.map(plot_dens_grad, range(1000))

    print("All slices processed in parallel.")
