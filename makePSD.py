import numpy as np
import matplotlib
matplotlib.use('Agg')  #display not required
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.signal import welch
import util_functions

packetfreq = 5000 #Hz
rhoinf = 0.03754  #kg/m^3
Tinf = 51         #K
ainf = 143.150    #m/s 
R = 287.0         #J/kg-K
Cv = 717          #J/kg-K
Lsep = 0.07       #m
Uinf = 859        #m/s
Pinf = rhoinf*R*Tinf

slices = []
tapnum = [700,900]

num_slices = 30000 #Number of slices per save file
filenames = ["../5000Hz_276mm/outputs/outputs_060000/taps_K151_060000"]

n_iter = 60000 - 30000
fs = 10e6  #Sampling frequency
nperseg = int(np.floor(n_iter / 4.5))
noverlap = int(np.floor(0.5*nperseg))

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


#Compute PSDs

plt.rcParams.update({'font.size': 16,'axes.labelsize': 16,
                     'axes.titlesize': 16,'xtick.labelsize': 16,
                     'ytick.labelsize': 16})

fig, axes = plt.subplots(3, 1, figsize=(8, 6))

axes[0].plot(slices[0]["X"][:].flatten(),slices[0]["Y"][:].flatten(), color='black')
axes[0].scatter(slices[0]["X"][tapnum[0]].flatten(), slices[0]["Y"][tapnum[0]].flatten(), color='magenta')
axes[0].scatter(slices[0]["X"][tapnum[1]].flatten(), slices[0]["Y"][tapnum[1]].flatten(), color='green')
axes[0].set_title("Tap Locations")
axes[0].set_xlabel("x [mm]")
axes[0].set_ylabel("y [mm]")
axes[0].set_xlim(370, 600)
axes[0].grid(True)

f, psd = welch(P[0]-np.mean(P[0]), fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
sigma = np.mean((P[0] - np.mean(P[0]))**2)
axes[1].semilogx(f*Lsep/Uinf, f*psd/sigma, color='magenta')
axes[1].axvline(x= packetfreq*Lsep/Uinf, linestyle='--', linewidth=1, label='Vertical line')
axes[1].axvline(x= 2*packetfreq*Lsep/Uinf, linestyle='--', linewidth=1, label='Vertical line')
axes[1].axvline(x= 3*packetfreq*Lsep/Uinf, linestyle='--', linewidth=1, label='Vertical line')
axes[1].axvline(x= 4*packetfreq*Lsep/Uinf, linestyle='--', linewidth=1, label='Vertical line')
axes[1].set_title("PSD | Separation")
axes[1].set_xlabel("$St_{L}$")
axes[1].set_ylabel("$f*G(f)/\sigma^{2}$")
axes[1].set_xlim(0.05,10)
axes[1].grid(True)

f, psd = welch(P[1]-np.mean(P[1]), fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
sigma = np.mean((P[1] - np.mean(P[1]))**2)
axes[2].semilogx(f*Lsep/Uinf, f*psd/sigma, color='green')
axes[2].axvline(x= packetfreq*Lsep/Uinf, linestyle='--', linewidth=1, label='Vertical line')
axes[2].axvline(x= 2*packetfreq*Lsep/Uinf, linestyle='--', linewidth=1, label='Vertical line')
axes[2].axvline(x= 3*packetfreq*Lsep/Uinf, linestyle='--', linewidth=1, label='Vertical line')
axes[2].axvline(x= 4*packetfreq*Lsep/Uinf, linestyle='--', linewidth=1, label='Vertical line')
axes[2].set_title("PSD | Mid-Bubble")
axes[2].set_xlabel("$St_{L}$")
axes[2].set_ylabel("$f*G(f)/\sigma^{2}$")
axes[2].set_xlim(0.05,10)
axes[2].grid(True)

plt.tight_layout()
plt.savefig('PSD.png', dpi=300)
plt.show()
