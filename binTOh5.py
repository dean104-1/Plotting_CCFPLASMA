import config as cfg
import util_functions
from multiprocessing import Pool


foldername = f"{cfg.basename}/outputs_0150000/"

filename_taps = [f"{foldername}/taps_K076_0150000",
                 f"{foldername}/taps_K151_0150000",
                 f"{foldername}/taps_K226_0150000",
                 f"{foldername}/taps_K301_0150000",
                 f"{foldername}/taps_K376_0150000",
                 f"{foldername}/taps_K451_0150000",
                 f"{foldername}/taps_K526_0150000"]

filename_slices = [f"{foldername}/slices_J1200_0150000",
                   f"{foldername}/slices_J1300_0150000",
                   f"{foldername}/slices_K001_0150000",
                   f"{foldername}/slices_K151_0150000",
                   f"{foldername}/slices_K301_0150000",
                   f"{foldername}/slices_K451_0150000",
                   f"{foldername}/slices_K601_0150000",
                   f"{foldername}/slices_wall_0150000"]

def loadtaps(nn):
    loadpath_taps = util_functions.loadslices_h5(filename_taps[nn],100000)
    

def loadslices(nn):
    loadpath_taps = util_functions.loadslices_h5(filename_slices[nn],5000)
    
    
if __name__ == "__main__":
    num_workers = 8

    with Pool(processes=num_workers) as pool:
        pool.map(loadtaps, range(len(filename_taps)))

    print("All taps processed in parallel.")
    
    
if __name__ == "__main__":
    num_workers = 8

    with Pool(processes=num_workers) as pool:
        pool.map(loadslices, range(len(filename_slices)))

    print("All taps processed in parallel.")