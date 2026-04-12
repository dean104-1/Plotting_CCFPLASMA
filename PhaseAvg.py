import config as cfg
import numpy as np
#matplotlib.use('Agg')  #display not required
import matplotlib.pyplot as plt
import util_functions
import h5py

period = 1/cfg.packetfreq
iter_interval = cfg.fsample/cfg.packetfreq

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
        loadP.append(hf['p'][:,cfg.tapnum,0,0])
        
P = np.concatenate(loadP, axis=0)
print('--Done Loading Data--')

nperiods = np.round(np.size(P, 0)/iter_interval).astype(int)

P_bins = np.zeros((nperiods,(np.sum(cfg.lengths_taps)/nperiods).astype(int)))
for ii in range(nperiods):
    startidx = int(ii*iter_interval)
    endidx = int((ii+1)*iter_interval)
    P_bins[ii,:] = P[startidx:endidx]

timevec = np.arange(1, iter_interval + 1) * cfg.timestep

P_bins_avg = np.mean(P_bins/cfg.Pinf, axis=0)
P_bins_std = np.std(P_bins/cfg.Pinf, axis=0)

plt.rcParams.update({'font.size': 16,'axes.labelsize': 16,
                     'axes.titlesize': 16,'xtick.labelsize': 16,
                     'ytick.labelsize': 16})

plt.plot(timevec/period,P_bins_avg, c='red', label="Phase-Average")
plt.plot(timevec/period,P_bins_avg+P_bins_std, c='lime')
plt.plot(timevec/period,P_bins_avg-P_bins_std, c='lime')
plt.scatter(timevec/period,P_bins[0,:]/cfg.Pinf, s=10, c='blue', label="Sampled Data")
for jj in range(P_bins.shape[0]):
    plt.scatter(timevec/period,P_bins[jj,:]/cfg.Pinf, s=10, c='blue')
plt.xlabel("$t/T_{packet}$")
plt.ylabel("$P/P_{\infty}$")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('PhaseAvg.png', dpi=300)
plt.show()