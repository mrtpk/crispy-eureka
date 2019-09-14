* The dataset is from KITTI road detection [benchmark](http://www.cvlibs.net/datasets/kitti/eval_road.php)
* The dataset is of two parts- road images and LiDAR data.
  * Image [dataset](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip)
  * LiDAR [dataset](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road_velodyne.zip)
* Python development [kit](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_road.zip)

### Generation of BEV view of ground truth from KITTI dataset:

`transform2BEV.py` from devkit is used to generate BEV of ground truth images. The environment configuration to run the script is in `pydevkit_env.yaml`. The Bird's eye view parameters are modified at `line 103` in `BirdsEyeView.py` to `bev_res= 0.1, bev_xRange_minMax = (-10, 10), bev_zRange_minMax = (6, 46)`. The `line 49` in `transform2BEV.py` has to modified to `fileList_data = glob(dataFiles + '/*.png')` to excute the script successfully. To run the script- `python transform2BEV.py [input folder] [calibration folder path] [output folder]`.

### Citation:

@INPROCEEDINGS{Fritsch2013ITSC,
  author = {Jannik Fritsch and Tobias Kuehnl and Andreas Geiger},
  title = {A New Performance Measure and Evaluation Benchmark for Road Detection Algorithms},
  booktitle = {International Conference on Intelligent Transportation Systems (ITSC)},
  year = {2013}
}
