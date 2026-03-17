#On carpenter do: module load python/3.10.4
import numpy as np
import matplotlib
#matplotlib.use('Agg')  #display not required
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.signal import welch
import util_functions

packetfreq = 1000 #Hz
period = 1/packetfreq
fsample = 2e6     #Hz
iter_interval = fsample/packetfreq
rhoinf = 0.03754  #kg/m^3
Tinf = 51         #K
ainf = 143.150    #m/s 
R = 287.0         #J/kg-K
Cv = 717          #J/kg-K
Lsep = 0.20       #m
Uinf = 859        #m/s
Pinf = rhoinf*R*Tinf

slices = []
tapnum = [640,1250,1730] #Only the last one is plotted

num_slices = 30000 #Number of slices per save file
filenames = ["../1000Hz_276mm/outputs/outputs_060000/taps_pulse_060000",
             "../1000Hz_276mm/outputs/outputs_090000/taps_pulse_090000",
             "../1000Hz_276mm/outputs/outputs_120000/taps_pulse_120000"]

P = []

for kk in range(len(tapnum)):

    rho_star = []
    rho_star_u_star = []
    rho_star_v_star = []
    rho_star_w_star = []
    rho_star_e_star = []

    for ii in range(len(filenames)):

        slices = util_functions.loadslices(filenames[ii],num_slices)

        for jj in range(num_slices):

            rho_star.append(slices[jj]["Q"][tapnum[kk],0,0,0])
            rho_star_u_star.append(slices[jj]["Q"][tapnum[kk],0,0,1])
            rho_star_v_star.append(slices[jj]["Q"][tapnum[kk],0,0,2])
            rho_star_w_star.append(slices[jj]["Q"][tapnum[kk],0,0,3])
            rho_star_e_star.append(slices[jj]["Q"][tapnum[kk],0,0,4])


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

P = np.array(P)
print(P.shape)

nperiods = round(len(P[0,:])/iter_interval)

P_bins = []
for ii in range(nperiods):
    startidx = int(ii*iter_interval)
    endidx = int((ii+1)*iter_interval)
    P_bins.append(P[2,startidx:endidx])


P_bins = np.array(P_bins)
print(P_bins.shape)

timevec = np.arange(1, iter_interval + 1) * 5e-7

P_bins_avg = np.mean(P_bins/Pinf, axis=0)

plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
})

plt.plot(timevec/period,P_bins_avg, c='red', label="Phase-Average")
plt.scatter(timevec/period,P_bins[0,:]/Pinf, s=10, c='blue', label="Sampled Data")
for jj in range(P_bins.shape[0]):
    plt.scatter(timevec/period,P_bins[jj,:]/Pinf, s=10, c='blue')
plt.xlabel("$t/T_{packet}$")
plt.ylabel("$P/P_{\infty}$")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('PhaseAvg.png', dpi=300)
plt.show()
