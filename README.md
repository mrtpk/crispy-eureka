# crispy-eureka (new readme, please go through completely!)
* Evaluating the effect of spatial resolution for road detection using Lidar point clouds, camera and geometrical features
* [VISAPP 2020](http://www.visapp.visigrapp.org/CallForPapers.aspx) Malta, Position Papers Paper Submission: **November 15, 2019** (good target for an intermediate version)

## French conferences
* [ESANN 2020](https://www.esann.org/) Bruges, Beglium **18 Nov 2019** (more DNN based paper)
* [ISPRS 2020](http://www.isprs2020-nice.com/) Nice, France, 3 February 2020, Deadline for abstracts & full papers: technical sessions and young investigators
* [RFIAP 2020, CAp 2020](https://www.sciencesconf.org/browse/conference/?confid=8780) Conférences jointes sur l'Apprentissage automatique - Reconnaissance des Formes, Image, Apprentissage et Perception

## Later deadlines (conferences)
* [CVPR 2020](http://cvpr2020.thecvf.com/) US : Nov 15 2019 (tough to get in though ;) given the short time)
* [ECCV 2020](https://eccv2020.eu/important-dates/) SEC, GLASGOW, March 2020 (looks good for a good submission)
* [IROS 2020](http://www.iros2020.org/) Tentative Paper Submission Deadline : March 1, 2020
* [EUSIPCO 2020](https://eusipco2020.org/calls/) Full paper submission: February 21, 2020
* [ICPR 2020](https://www.micc.unifi.it/icpr2020/index.php/important-dates/) Main Conference June 1, 2020
* [3DV 2020](http://3dv2020.dgcv.nii.ac.jp/) International Conference on 3D Vision : early August, 2020

# Current model architectures (Front View)
- X is input domain size (WxHxK), S is binary segmentation 
- X64 \in 64x1000, X32 \in 32x1000, X16 \in 16x1000
## Method 1 : Predict in GT-64
- Here each network maps the input domain to the full resolution ground truth
- ```DNN 64 : X64 -> Enc64 -> z64 -> Dec64 -> S ```
- ```DNN 32 : X32 -> Enc32 -> z32 -> Dec32 -> S``` 
- ```DNN 16 : X16 -> Enc16 -> z16 -> Dec16 -> S``` 
- Here latent space variables are different sizes, i.e z16 < z32 < z64 since the inputs X's are of different sizes
- The quality of subsampling is confounded with the ground truth predicition quality
## Method 2 : Input X is always upsampled to 64x1000 and then Predict in GT-64
- ```DNN 64 : X64 -> Enc64 -> z64 -> Dec64 -> S ```
- ```DNN 32 : X32 -> Up2 -> Enc32 -> z32 -> Dec32 -> S``` 
- ```DNN 16 : X16 -> Up4 -> Enc16 -> z16 -> Dec16 -> S```
- Here the latent space variables z64, z32, z16 are all the same sizes since the inputs are upsampled always to the same sizes
- Upsamping can be fixed by using simple binary interpolation, or this can be a learnt layer, in which case seperating errors of reconstruction and classification/seg becomes a problem.
- We could introduce an auxilliary loss for upsampling to ensure the upsampling remains correct.
## Method 3 : Shared Multiscale FCN
- ``` DNN 64 : X64 -> [Enc -> z64 -> Dec] -> S ```
- ``` DNN 32 : X32 -> Up2 -> [Enc -> z64 -> Dec] -> S ```
- ``` DNN 16 : X16 -> Up4 -> [Enc -> z64 -> Dec] -> S ```
- The model [Enc -> z64 -> Dec] is shared across scales are we train the same network to perfor predictions at different levels of subsampling. We still have to ensure the upsampling loss is low.
- An alternative would be to aggregate the features from each upsampled-subsampled scale into a multiscale context module as in FPN/LoDNN
- Final architecture DNN   :
X64 -> Enc64 -> z64   
X32 -> Up2 -> Enc32 -> z32  
X16 -> Up4 -> Enc16 -> z16  
concat(z64, z32, z16)->Dec->S    
- This architecture is very similar to the Feature Pyramid Network used in Semantic Segmentation in images [FPN](https://towardsdatascience.com/review-fpn-feature-pyramid-network-object-detection-262fc7482610) [PSPNet](https://towardsdatascience.com/review-pspnet-winner-in-ilsvrc-2016-semantic-segmentation-scene-parsing-e089e5df177d)
- Pyramid Pooling Network Architecture:
![](`https://miro.medium.com/max/1304/1*IxUlWP8RBtxNS1N6hyBAxA.png`)
## Variational Loss
- In method 2 and 3 since the latent space vectors are the same size, a KL-divergence loss can be introduce to ensure that the subsampled feature maps are embedded to the same neighbourhood.
- This also provides a way to measure the likelihood of the input data samples/pointclouds.
- KL(z32||z64), KL(z16||z64) for the either the independent models or shared models.
## Multiscale decomposition of features
- If we look at the features in Front view or BEV, after subsampling the feature maps such as z_min, z_max, z_mean are monotonically decreasing since at each subsampling step we remove points. 
- This creates implies that z_min_64 will strict contain or equal to the z_min_32 and z_min_16 i.e. ```z_min_64 >= z_min_32 => z_min_64```
- One can now perform a finer decomposition of this feature map by consider classical multiscale transformations such as the wavelet transform which provides maps at different scales.
- This also demonstrates that the quality of the feature maps should consistently degrade, though the features extracted by the CNN for road segmentation might not the same as for reconstruction.
# New Goals :
- [ ] Understand why classical-64 is performing worse compared to classical 32
- [ ]Add data-generator for flow_from_directory [tutorial](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)
- Front view 
	- [ ] with SqueezeNet, 
	- [ ] with DarkNet ?
- BEV
	- [ ] with Unet ?
- Evaluate better data-augmentation 
	- Shadows by introducing objects
- Other data-aug : noise, peturbations, adding closings, or hole closings with morphological filtering
- Upload a few video outputs to demonstrate the output

# Alternative datasets
- Use Carla pointclouds generated using 64 layers, 32 layers, 16 layers, 8 layers(these are directly configurable in CARLA and would not require subsampling, though will be costly to store). The ground truth can be generated by mapping points into semantic segmentation ground truth provided by Carla.


# Contributions : 
- Geometrical features improve and dont degrage performance for road seg with subsampling
- Spectral features dont degrade performance for road seg with subsampling
- Evaluating the effect of density variation in classification of road segmentation
- First benchmark on Semantic KITTI for road segmentation
- Transfer learning to other datasets ? [Perform prediction and training on a dataset that is different from KITTI's sensor configuration ?]

# Future goals (Deadline to be fixed, CVPR ?)
- Conferences : 
	- [http://www.visapp.visigrapp.org](VISAPP 2020, Malta), October 4, 2019 (This seems ideal for a small version of the current paper)
	- [http://cvpr2020.thecvf.com/](CVPR 2020, Seattle, US), Nov 15, 2019 (this is quite a tough conference, would be great to aim for this, but we would need a solid submission)
	- [https://eccv2020.eu/important-dates/](ECCV 2020, Glasgow), March 2020 (TBA, this is quite a tough conference, would be great to aim for this, but we would need a solid submission)
	- [https://www.micc.unifi.it/icpr2020/index.php/important-dates/](ICPR, Milan) March 2, 2020 

- Work on Front view and evaluate performance with subsampling, models : u-net, squeezenet, lodnn (Leo/Ravi)
- Correct/Fix the HOG computation (Ravi), What is better way to perform fusion with image in front view and bird eye view and evaluate how this scales with subsampling
- Handle Class imbalance (very few points corresponding to road)
- **Why not perform complete semantic segmentation on all classes ? And study the performance on class vs subsampling ?**
	-(this objects would suffer with subsampling though camera fusion would help better here)

# Secondary goals if we have time
- **Density estimation** : Improve 2d-grid feature extraction in BEV using kernel density estimate
	- Kernel Density Filtering for Noisy Point Clouds in One Step [pdf](http://www.csd.uwo.ca/faculty/beau/PAPERS/imvip-15.pdf)
- **Feature Selection** Autoencoders to perform feature selection [Concrete Autoencoders](https://github.com/mfbalin/Concrete-Autoencoders)
- **Data-augmentation** to handle the density variation and shadows

