timestep = 1e-7       #s
packetfreq = 5000     #Hz
fsample = 1/timestep  #Hz

#Define thermo variables
rhoinf = 0.03754  #kg/m^3
Tinf = 51         #K
ainf = 143.150    #m/s
R = 287.0         #J/kg-K
Cv = 717          #J/kg-K
Uinf = 859        #m/s
Pinf = rhoinf*R*Tinf

num_slices = 30000 #Number of slices per save file

basename = "../5000Hz_276mm/outputs"
n_iter = 60000 - 30000
