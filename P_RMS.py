import config as cfg
import numpy as np
import matplotlib
#matplotlib.use('Agg')  #display not required
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.signal import welch
import util_functions

slices = []

filenames = [f"{cfg.basename}/outputs_060000/taps_K151_060000"]
             #"../5000Hz_276mm/outputs/outputs_090000/taps_pulse_090000",
             #"../5000Hz_276mm/outputs/outputs_120000/taps_pulse_120000"]

P = []

rho_star = []
rho_star_u_star = []
rho_star_v_star = []
rho_star_w_star = []
rho_star_e_star = []

for ii in range(len(filenames)):

    slices = util_functions.loadslices(filenames[ii],cfg.num_slices)

    for jj in range(cfg.num_slices):

        rho_star.append(slices[jj]["Q"][:,0,0,0])
        rho_star_u_star.append(slices[jj]["Q"][:,0,0,1])
        rho_star_v_star.append(slices[jj]["Q"][:,0,0,2])
        rho_star_w_star.append(slices[jj]["Q"][:,0,0,3])
        rho_star_e_star.append(slices[jj]["Q"][:,0,0,4])


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

P_avg = np.mean(P, axis=1) #take average over time
P_prime = P - P_avg
P_RMS = np.sqrt(np.mean(P_prime**2, axis=1))
P_RMS = P_RMS.squeeze()   #Flatten the array


#Plot P_RMS

radius = np.sqrt(slices[0]["Y"][:].flatten()**2 + slices[0]["Z"][:].flatten()**2)

plt.rcParams.update({'font.size': 16,'axes.labelsize': 16,
                     'axes.titlesize': 16,'xtick.labelsize': 16,
                     'ytick.labelsize': 16})

fig, ax1 = plt.subplots(figsize=(8,4))

ax1.plot(slices[0]["X"][:].flatten(), P_RMS/cfg.Pinf, 'blue')
ax1.set_xlabel('x [mm]')
ax1.set_ylabel("$P'_{RMS}/P_{\infty}$", color='blue')
ax1.set_xlim(350,600)
ax1.set_ylim(0,2.3)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, color='black')

ax2 = ax1.twinx()   # second y-axis
ax2.plot(slices[0]["X"][:].flatten(), radius, 'black')
ax2.set_ylabel('Body Radius [mm]', color='black')
ax2.set_xlim(350,600)
ax2.set_ylim(20,100)
ax2.tick_params(axis='y', labelcolor='black')

plt.tight_layout()
plt.savefig('P_RMS.png', dpi=300)
plt.show()

