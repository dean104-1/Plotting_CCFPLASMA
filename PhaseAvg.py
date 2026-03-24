import config as cfg
import numpy as np
import matplotlib
#matplotlib.use('Agg')  #display not required
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.signal import welch
import util_functions

period = 1/cfg.packetfreq
iter_interval = cfg.fsample/cfg.packetfreq

slices = []

P = []

rho_star = []
rho_star_u_star = []
rho_star_v_star = []
rho_star_w_star = []
rho_star_e_star = []

for ii in range(len(cfg.filenames_taps)):

    slices = util_functions.loadslices(cfg.filenames_taps[ii],cfg.lengths_taps[ii])

    for nn in range(cfg.lengths_taps[ii]):

        rho_star.append(slices[nn]["Q"][cfg.tapnum,0,0,0])
        rho_star_u_star.append(slices[nn]["Q"][cfg.tapnum,0,0,1])
        rho_star_v_star.append(slices[nn]["Q"][cfg.tapnum,0,0,2])
        rho_star_w_star.append(slices[nn]["Q"][cfg.tapnum,0,0,3])
        rho_star_e_star.append(slices[nn]["Q"][cfg.tapnum,0,0,4])

rho = np.array(rho_star)*cfg.rhoinf
u_star = np.array(rho_star_u_star)/np.array(rho_star)
u = u_star*cfg.ainf
v_star = np.array(rho_star_v_star)/np.array(rho_star)
v = v_star*cfg.ainf
w_star = np.array(rho_star_w_star)/np.array(rho_star)
w = w_star*cfg.ainf
e_star = np.array(rho_star_e_star)/np.array(rho_star)
e = e_star*cfg.ainf**2
temp = (e - 0.5*(u**2 + v**2 + w**2))/cfg.Cv

P.append(rho*cfg.R*temp)
P = np.array(P)

P = P.flatten()
print(P.shape)

nperiods = round(len(P[:])/iter_interval)

P_bins = []
for ii in range(nperiods):
    startidx = int(ii*iter_interval)
    endidx = int((ii+1)*iter_interval)
    P_bins.append(P[startidx:endidx])

P_bins = np.array(P_bins)
print(P_bins.shape)

timevec = np.arange(1, iter_interval + 1) * cfg.timestep

P_bins_avg = np.mean(P_bins/cfg.Pinf, axis=0)

plt.rcParams.update({'font.size': 16,'axes.labelsize': 16,
                     'axes.titlesize': 16,'xtick.labelsize': 16,
                     'ytick.labelsize': 16})

plt.plot(timevec/period,P_bins_avg, c='red', label="Phase-Average")
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
