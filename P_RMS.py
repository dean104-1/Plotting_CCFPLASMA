import config as cfg
import numpy as np
#matplotlib.use('Agg')  #display not required
import matplotlib.pyplot as plt
import util_functions
import h5py

loadP = []
for ii in range(len(cfg.filenames_taps)):
    loadpath = util_functions.loadslices_h5(cfg.filenames_taps[ii],cfg.lengths_taps[ii])
    with h5py.File(loadpath, 'r') as hf:
        XLOC = hf['X'][:,:,:]
        YLOC = hf['Y'][:,:,:]
        ZLOC = hf['Z'][:,:,:]
        loadP.append(hf['p'][:,:,:,:])
        
P = np.concatenate(loadP, axis=0)
print('--Done Loading Data--')

P_avg = np.mean(P, axis=0) #take average over time
P_prime = P - P_avg
P_RMS = np.sqrt(np.mean(P_prime**2, axis=0))


#Plot P'_RMS

radius = np.sqrt(YLOC**2 + ZLOC**2)

plt.rcParams.update({'font.size': 16,'axes.labelsize': 16,
                     'axes.titlesize': 16,'xtick.labelsize': 16,
                     'ytick.labelsize': 16})

fig, ax1 = plt.subplots(figsize=(8,4))

ax1.plot(XLOC.flatten(), (P_RMS/cfg.Pinf).flatten(), 'blue')
ax1.set_xlabel('x [mm]')
ax1.set_ylabel("$P'_{RMS}/P_{\infty}$", color='blue')
ax1.set_xlim(350,600)
ax1.set_ylim(0,2.3)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, color='black')

ax2 = ax1.twinx()   # second y-axis
ax2.plot(XLOC.flatten(), radius.flatten(), 'black')
ax2.set_ylabel('Body Radius [mm]', color='black')
ax2.set_xlim(350,600)
ax2.set_ylim(20,100)
ax2.tick_params(axis='y', labelcolor='black')

plt.tight_layout()
plt.savefig('P_RMS.png', dpi=300)
plt.show()

