packetfreq = 5000     #Hz
basename = "../5000Hz_60p/outputs"

tapnum = 700     #J index of the tap you's like to plot for PhaseAvg.py
tapnum_vec = [700,900]   #J index vector of the taps to plot for makePSD and makeSlices

dofilter = 1  #Apply gaussian filter for shearTrack.py

Lsep = 0.07

filenames_slices = [f"{basename}/outputs_0050000/slices_K151_0050000"]
lengths_slices = [2000]

filenames_taps = [f"{basename}/outputs_0050000/taps_K151_0050000"]
lengths_taps = [40000]

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
