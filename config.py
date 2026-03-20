packetfreq = 5000     #Hz
basename = "../5000Hz_70p/outputs"

num_taps = 30000 #Number of taps per save file
num_slices = 1500 #Number of slices per save file (or as many as you'd like to plot)
tapnum = 700     #J index of the tap you's like to plot for PhaseAvg.py
tapnum_vec = [700,900]   #J index vector of the taps to plot for makePSD and makeSlices

dofilter = 1  #Apply gaussian filter for shearTrack.py

n_iter = 60000 - 30000

Lsep = 0.07

filenames_slices = [f"{basename}/outputs_0060000/slice_K151_0060000",
                    f"{basename}/outputs_0090000/slice_K151_0090000"]

lengths_slices = [1500, 1500]

#Define thermo variables
rhoinf = 0.03754  #kg/m^3
Tinf = 51         #K
ainf = 143.150    #m/s
R = 287.0         #J/kg-K
Cv = 717          #J/kg-K
Uinf = 859        #m/s
Pinf = rhoinf*R*Tinf

timestep = 1e-7       #s
fsample = 1/timestep  #Hz
