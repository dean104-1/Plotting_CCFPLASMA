import os
os.environ['VTK_GRAPHICS_BACKEND'] = 'OSMesa'
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['PYVISTA_USE_OSMESA'] = 'true'

import config as cfg
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')  #display not required
import matplotlib.pyplot as plt
from multiprocessing import Pool
import util_functions
import pyvista as pv
import colorcet as cc
import h5py


filename_slices = f"{cfg.basename}/outputs_0100000/slices_0100000"

loadpath_slices = util_functions.loadslices_h5(filename_slices,1000)
with h5py.File(loadpath_slices, 'r') as hf:
    NJ = hf.attrs['NJ']
    NK = hf.attrs['NK']
    NL = hf.attrs['NL']
    XLOC = hf['X'][:,:,:]
    YLOC = hf['Y'][:,:,:]
    ZLOC = hf['Z'][:,:,:]
    rho = hf['rho'][:,:,:,:]

print('--Done Loading Data--')

if(cfg.dofilter==1):
    #for jj in range(slices[0]["NJ"]):
    #    for ll in range(slices[0]["NL"]):
    #        density[:,jj,ll] = ndimage.gaussian_filter(density[:,jj,ll], sigma=1)
    for nn in range(1000):
        rho[nn,:,:,:] = ndimage.gaussian_filter(rho[nn,:,:,:], sigma=3)

def plot_shear_track(nn):
    density_data = rho[nn,:,:,:]/cfg.rhoinf
    
    radius = np.sqrt(YLOC**2 + ZLOC**2)

    grid = pv.StructuredGrid(XLOC, radius, np.zeros_like(XLOC))
    grid["rho_plot"] = density_data.flatten(order="F")

    contour1 = grid.contour([0.7])
    points1 = contour1.points

    contour2 = grid.contour([0.5])
    points2 = contour2.points
    
    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = "white"
    plotter.add_mesh(grid, scalars="rho_plot", opacity=1.0, clim=[0.0,2.0], lighting=False, smooth_shading=False, cmap="CET_L1",show_scalar_bar=False)  #CET_CBTL3
    #plotter.add_scalar_bar(title=r'$\rho/\rho_{\infty}$', n_labels=6, title_font_size=30, label_font_size=30, color="black", position_x=0.04, position_y=0.90)
    plotter.add_mesh(contour1, color="red", line_width=4)
    plotter.add_mesh(contour2, color="lime", line_width=4)
    #plotter.view_xy()
    plotter.camera_position = [(470.0, 50.0, 130.0), (470.0, 50.0, 0.0), (0.0, 1.0, 0.0)]
    plotter.camera.zoom(1.0)
    plotter.window_size = (6000,2400)
    plotter.enable_parallel_projection()
    plotter.screenshot(f"images/myplot_{nn:02d}.png")
    plotter.close()
    #plotter.show(screenshot=f"images/myplot_{nn:02d}.png")

if __name__ == "__main__":
    num_workers = 8

    with Pool(processes=num_workers) as pool:
        #pool.map(plot_tap_history, range(cfg.num_slices))
        pool.map(plot_shear_track, range(400))

        print("All slices processed in parallel.")
