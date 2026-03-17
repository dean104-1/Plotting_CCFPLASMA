import numpy as np
import matplotlib.pyplot as plt
import util_functions

slices = []
#total_e = []

num_slices = 10
filename = "BC_201.1.7"

slices = util_functions.loadslices(filename,num_slices)

rho_inf = 0.03754
a_inf = 143.15
cellVol = (.01/80)*(.01/80)*(.1/800)

for jj in range(num_slices):

    rho_vol = rho_inf*cellVol  #Assumes that density and cell volume are constant throughout domain
    total_e = np.sum((a_inf**2)*slices[jj]["Q"][:,:,:,4]*rho_vol)
    print(total_e, slices[jj]["ISTEP"])

    #print(slices[jj]["Q"][700,40,40,0])


