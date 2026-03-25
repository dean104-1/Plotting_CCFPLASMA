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

slices = []

filename_wall = f"{cfg.basename}/outputs_0050000/slices_wall_0050000" 
filename_right_wall = f"{cfg.basename}/outputs_0050000/slices_K601_0050000"
filename_cross = f"{cfg.basename}/outputs_0050000/slices_J1300_0050000" 

slices_wall = util_functions.loadslices(filename_wall,cfg.lengths_slices[0])
slices_right_wall = util_functions.loadslices(filename_right_wall,cfg.lengths_slices[0])
slices_cross = util_functions.loadslices(filename_cross,cfg.lengths_slices[0])

print("Done loading data")

def plot_dens_grad(nn):
    slice_data_wall = slices_wall[nn]
    slice_data_right_wall = slices_right_wall[nn]
    slice_data_cross = slices_cross[nn]
    
    grid_wall = pv.StructuredGrid(slice_data_wall["X"][:,:,:], slice_data_wall["Y"][:,:,:], slice_data_wall["Z"][:,:,:])
    grid_wall["rho"] = slice_data_wall["Q"][:,:,:,0].flatten(order="F")
    edges_wall = grid_wall.extract_feature_edges()

    grid_right_wall = pv.StructuredGrid(slice_data_right_wall["X"][:,:,:], slice_data_right_wall["Y"][:,:,:], slice_data_right_wall["Z"][:,:,:])
    grid_right_wall["rho"] = slice_data_right_wall["Q"][:,:,:,0].flatten(order="F")
    edges_right_wall = grid_right_wall.extract_feature_edges()

    grid_cross = pv.StructuredGrid(slice_data_cross["X"][:,:,:], slice_data_cross["Y"][:,:,:], slice_data_cross["Z"][:,:,:])
    grid_cross["rho"] = slice_data_cross["Q"][:,:,:,0].flatten(order="F")
    edges_cross = grid_cross.extract_feature_edges()

    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = "white"
    plotter.add_mesh(grid_wall, scalars="rho", opacity=1.0, clim=[0.0,2.0], lighting=False, smooth_shading=False, cmap="hot",show_scalar_bar=False)
    plotter.add_mesh(grid_right_wall, scalars="rho", opacity=1.0, clim=[0.0,6.0], lighting=False, smooth_shading=False, cmap="CET_L1",show_scalar_bar=False)
    plotter.add_mesh(grid_cross, scalars="rho", opacity=0.75, clim=[0.0,6.0], lighting=False, smooth_shading=False, cmap="CET_L1",show_scalar_bar=False)

    plotter.add_mesh(edges_wall, color="black", line_width=8)
    plotter.add_mesh(edges_right_wall, color="black", line_width=8)
    plotter.add_mesh(edges_cross, color="black", line_width=8)

    #plotter.add_scalar_bar(title=r'$\rho/\rho_{\infty}$', n_labels=6, title_font_size=30, label_font_size=30, color="black", position_x=0.04, position_y=0.90)
    #plotter.view_xy()
    plotter.camera.zoom(1.2)
    plotter.camera_position = [(280.0, 100.0, 250.0), (480.0, 50.0, 50.0), (0.0, 1.0, 0.0)]
    plotter.window_size = (5000,2400)
    plotter.enable_parallel_projection()
    plotter.screenshot(f"images/myplot_{nn:02d}.png")
    plotter.close()

if __name__ == "__main__":
    num_workers = 32

    with Pool(processes=num_workers) as pool:
        pool.map(plot_dens_grad, range(cfg.lengths_slices[0]))

        print("All slices processed in parallel.")
