# Helper functions to load the data for experiments
# load_bin_file: loads point cloud stored in binary file
# load_np_file: read the numpy file into memory
# read KITTI calibration files
# func to load the gt from KITTI and LODNN
# load the images
import numpy as np
def load_bin_file(bin_path, n_cols=4):
    '''
    Load a binary file and convert it into a numpy array.
    '''
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, n_cols)

