import config as cfg
import numpy as np
import matplotlib
matplotlib.use('Agg')  #display not required
import matplotlib.pyplot as plt
from scipy.signal import welch
import util_functions
import h5py

slices = []

nperseg = int(np.floor(np.sum(cfg.lengths_taps) / 4.5))
noverlap = int(np.floor(0.5*nperseg))

loadP = []
for ii in range(len(cfg.filenames_taps)):
    loadpath = util_functions.loadslices_h5(cfg.filenames_taps[ii],cfg.lengths_taps[ii])
    with h5py.File(loadpath, 'r') as hf:
        NJ = hf.attrs['NJ']
        NK = hf.attrs['NK']
        NL = hf.attrs['NL']
        XLOC = hf['X'][:,:,:]
        YLOC = hf['Y'][:,:,:]
        ZLOC = hf['Z'][:,:,:]
        loadP.append(hf['p'][:,cfg.tapnum_vec,0,0])
        
P = np.concatenate(loadP, axis=0)
print('--Done Loading Data--')


#Compute PSDs

radius = np.sqrt(YLOC**2 + ZLOC**2)
radius_tap = np.zeros(2)
radius_tap[0] = np.sqrt(YLOC[cfg.tapnum_vec[0],:,:]**2 + ZLOC[cfg.tapnum_vec[0],:,:]**2).item()
radius_tap[1] = np.sqrt(YLOC[cfg.tapnum_vec[1],:,:]**2 + ZLOC[cfg.tapnum_vec[1],:,:]**2).item()

plt.rcParams.update({'font.size': 16,'axes.labelsize': 16,
                     'axes.titlesize': 16,'xtick.labelsize': 16,
                     'ytick.labelsize': 16})

fig, axes = plt.subplots(3, 1, figsize=(8, 6))

axes[0].plot(XLOC.flatten(), radius.flatten(), color='black')
axes[0].scatter(XLOC[cfg.tapnum_vec[0],:,:].flatten(), radius_tap[0], color='magenta')
axes[0].scatter(XLOC[cfg.tapnum_vec[1],:,:].flatten(), radius_tap[1], color='green')
axes[0].set_title("Tap Locations")
axes[0].set_xlabel("x [mm]")
axes[0].set_ylabel("y [mm]")
axes[0].set_xlim(370, 600)
axes[0].grid(True)

f, psd = welch(P[:,0]-np.mean(P[:,0]), fs=cfg.fsample, nperseg=nperseg, noverlap=noverlap, window='hann')
sigma = np.mean((P[:,0] - np.mean(P[:,0]))**2)
axes[1].semilogx(f*cfg.Lsep/cfg.Uinf, f*psd/sigma, color='magenta')
axes[1].axvline(x= cfg.packetfreq*cfg.Lsep/cfg.Uinf, linestyle='--', linewidth=1, label='Vertical line')
axes[1].axvline(x= 2*cfg.packetfreq*cfg.Lsep/cfg.Uinf, linestyle='--', linewidth=1, label='Vertical line')
axes[1].axvline(x= 3*cfg.packetfreq*cfg.Lsep/cfg.Uinf, linestyle='--', linewidth=1, label='Vertical line')
axes[1].axvline(x= 4*cfg.packetfreq*cfg.Lsep/cfg.Uinf, linestyle='--', linewidth=1, label='Vertical line')
axes[1].set_title("PSD | Separation")
axes[1].set_xlabel("$St_{L}$")
axes[1].set_ylabel("$f*G(f)/\sigma^{2}$")
axes[1].set_xlim(0.01,1000)
axes[1].grid(True)

f, psd = welch(P[:,1]-np.mean(P[:,1]), fs=cfg.fsample, nperseg=nperseg, noverlap=noverlap, window='hann')
sigma = np.mean((P[:,1] - np.mean(P[:,1]))**2)
axes[2].semilogx(f*cfg.Lsep/cfg.Uinf, f*psd/sigma, color='green')
axes[2].axvline(x= cfg.packetfreq*cfg.Lsep/cfg.Uinf, linestyle='--', linewidth=1, label='Vertical line')
axes[2].axvline(x= 2*cfg.packetfreq*cfg.Lsep/cfg.Uinf, linestyle='--', linewidth=1, label='Vertical line')
axes[2].axvline(x= 3*cfg.packetfreq*cfg.Lsep/cfg.Uinf, linestyle='--', linewidth=1, label='Vertical line')
axes[2].axvline(x= 4*cfg.packetfreq*cfg.Lsep/cfg.Uinf, linestyle='--', linewidth=1, label='Vertical line')
axes[2].set_title("PSD | Mid-Bubble")
axes[2].set_xlabel("$St_{L}$")
axes[2].set_ylabel("$f*G(f)/\sigma^{2}$")
axes[2].set_xlim(0.01,1000)
axes[2].grid(True)

plt.tight_layout()
plt.savefig('PSD.png', dpi=300)
plt.show()
