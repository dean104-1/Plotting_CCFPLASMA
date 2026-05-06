#On carpenter do: module load cray-python
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

filename_wall = f"{cfg.basename}/outputs_0060000/slices_wall_0060000" 
filename_right_wall = f"{cfg.basename}/outputs_0060000/slices_K151_0060000"
#filename_cross = f"{cfg.basename}/outputs_0060000/slices_J1300_0060000"

loadpath_wall = util_functions.loadslices_h5(filename_wall,1000)
with h5py.File(loadpath_wall, 'r') as hf:
    NJ_wall = hf.attrs['NJ']
    NK_wall = hf.attrs['NK']
    NL_wall = hf.attrs['NL']
    XLOC_wall = hf['X'][:,:,:]
    YLOC_wall = hf['Y'][:,:,:]
    ZLOC_wall = hf['Z'][:,:,:]
    rho_wall = hf['rho'][:,:,:,:]

loadpath_right_wall = util_functions.loadslices_h5(filename_right_wall,1000)
with h5py.File(loadpath_right_wall, 'r') as hf:
    NJ_right_wall = hf.attrs['NJ']
    NK_right_wall = hf.attrs['NK']
    NL_right_wall = hf.attrs['NL']
    XLOC_right_wall = hf['X'][:,:,:]
    YLOC_right_wall = hf['Y'][:,:,:]
    ZLOC_right_wall = hf['Z'][:,:,:]
    rho_right_wall = hf['rho'][:,:,:,:]
    
# loadpath_cross = util_functions.loadslices_h5(filename_cross,1000)
# with h5py.File(loadpath_cross, 'r') as hf:
#     NJ_cross = hf.attrs['NJ']
#     NK_cross = hf.attrs['NK']
#     NL_cross = hf.attrs['NL']
#     XLOC_cross = hf['X'][:,:,:]
#     YLOC_cross = hf['Y'][:,:,:]
#     ZLOC_cross = hf['Z'][:,:,:]
#     rho_cross = hf['rho'][:,:,:,:]

print("Done loading data")

def plot_dens(nn):
    slice_data_wall = rho_wall[nn,:,:,:]/cfg.rhoinf
    slice_data_right_wall = rho_right_wall[nn,:,:,:]/cfg.rhoinf
    # slice_data_cross = rho_cross[nn,:,:,:]/cfg.rhoinf
    
    grid_wall = pv.StructuredGrid(XLOC_wall, YLOC_wall, ZLOC_wall)
    grid_wall["rho"] = slice_data_wall.flatten(order="F")
    edges_wall = grid_wall.extract_feature_edges()
    
    grid_right_wall = pv.StructuredGrid(XLOC_right_wall, YLOC_right_wall, ZLOC_right_wall)
    grid_right_wall["rho"] = slice_data_right_wall.flatten(order="F")
    edges_right_wall = grid_right_wall.extract_feature_edges()
    
    # grid_cross = pv.StructuredGrid(XLOC_cross, YLOC_cross, ZLOC_cross)
    # grid_cross["rho"] = slice_data_cross.flatten(order="F")
    # edges_cross = grid_cross.extract_feature_edges()

    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = "white"
    plotter.add_mesh(grid_wall, scalars="rho", opacity=1.0, clim=[0.0,2.0], lighting=False, smooth_shading=False, cmap="hot",show_scalar_bar=False)
    plotter.add_mesh(grid_right_wall, scalars="rho", opacity=1.0, clim=[0.0,6.0], lighting=False, smooth_shading=False, cmap="CET_L1",show_scalar_bar=False)
    # plotter.add_mesh(grid_cross, scalars="rho", opacity=0.75, clim=[0.0,6.0], lighting=False, smooth_shading=False, cmap="CET_L1",show_scalar_bar=False)

    plotter.add_mesh(edges_wall, color="black", line_width=8)
    plotter.add_mesh(edges_right_wall, color="black", line_width=8)
    # plotter.add_mesh(edges_cross, color="black", line_width=8)

    #plotter.add_scalar_bar(title=r'$\rho/\rho_{\infty}$', n_labels=6, title_font_size=30, label_font_size=30, color="black", position_x=0.04, position_y=0.90)
    #plotter.view_xy()
    plotter.camera.zoom(1.0)
    plotter.camera_position = [(280.0, 100.0, 250.0), (470.0, 25.0, 0.0), (0.0, 1.0, 0.0)]
    plotter.window_size = (5000,2400)
    plotter.enable_parallel_projection()
    plotter.screenshot(f"images/myplot_{nn:02d}.png")
    plotter.close()

if __name__ == "__main__":
    num_workers = 32

    with Pool(processes=num_workers) as pool:
        #pool.map(plot_dens, range(cfg.lengths_slices[0]))
        pool.map(plot_dens, range(1000))
        print("All slices processed in parallel.")
