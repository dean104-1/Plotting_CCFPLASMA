#On carpenter do: module load cray-python
import numpy as np
import matplotlib
matplotlib.use('Agg')  #display not required
import matplotlib.pyplot as plt
from multiprocessing import Pool
import util_functions

slices_wall = []
slices_right_wall = []
slices_left_wall = []

timestep = 1e-7
num_slices = 400
filename_wall = "../5000Hz_276mm/outputs/outputs_020000/wall_020000"
filename_right_wall = "../5000Hz_276mm/outputs/outputs_020000/right_wall_020000"
filename_left_wall = "../5000Hz_276mm/outputs/outputs_020000/left_wall_020000"

slices_wall = util_functions.loadslices(filename_wall,num_slices)
slices_right_wall = util_functions.loadslices(filename_right_wall,num_slices)
slices_left_wall = util_functions.loadslices(filename_left_wall,num_slices)

print("Done loading data")

def anFullBody(nn):
    slices_data_wall = slices_wall[nn]
    slices_data_right_wall = slices_right_wall[nn]
    slices_data_left_wall = slices_left_wall[nn]

    y_rotated_wall = slices_data_wall["Y"][:,:,0]*np.cos(np.radians(45)) - slices_data_wall["Z"][:,:,0]*np.sin(np.radians(45)) 
    y_rotated_right_wall = slices_data_right_wall["Y"][:,0,:]*np.cos(np.radians(45)) - slices_data_right_wall["Z"][:,0,:]*np.sin(np.radians(45))
    y_rotated_left_wall = slices_data_left_wall["Y"][:,0,:]*np.cos(np.radians(45)) - slices_data_left_wall["Z"][:,0,:]*np.sin(np.radians(45))

    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16
    })

    fig, ax = plt.subplots(figsize=(18, 8.64))
    ax.contourf(slices_data_wall["X"][:,:,0], y_rotated_wall, slices_data_wall["Q"][:,:,0,0] , np.linspace(0, 1.8, 300, endpoint=True), cmap='hot', extend='both')
    ax.contourf(slices_data_right_wall["X"][:,0,:], y_rotated_right_wall, slices_data_right_wall["Q"][:,0,:,0] , np.linspace(0, 2.5, 300, endpoint=True), cmap='gray', extend='both')        
    ax.contourf(slices_data_left_wall["X"][:,0,:], y_rotated_left_wall, slices_data_left_wall["Q"][:,0,:,0] , np.linspace(0, 2.5, 300, endpoint=True), cmap='gray', extend='both')

    plt.xlim(350, 600)
    plt.ylim(-60, 60)
    plt.title(r'$\rho/\rho_{\infty}$')
    plt.xlabel('x [mm]')
    plt.ylabel(r'span [mm]')
    plt.savefig(f'images/myplot_{nn:02d}.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    num_workers = 32

    with Pool(processes=num_workers) as pool:
        #pool.map(plot_tap_history, range(num_slices))
        pool.map(anFullBody, range(num_slices))

    print("All slices processed in parallel.")

