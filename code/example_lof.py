from helpers import data_loaders as dls
import numpy as np
import pyvista as pv
from sklearn.neighbors import LocalOutlierFactor as LOF

pc_path = "../dataset/KITTI/dataset/data_road_velodyne/training/velodyne/umm_000035.bin"
points = dls.load_bin_file(pc_path)
point_cloud = pv.PolyData(points[:,:-1])                                                                                                                                                

clf = LOF(n_neighbors=100, contamination=0.1)                                                                                                                                                              
y_pred = clf.fit_predict(points) 


point_cloud["OutlierFactor"] = y_pred
point_cloud.plot(render_points_as_spheres=True)