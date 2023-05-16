# PH Dimension experiments
This repository contains some codes for calculating the PH Dim of different sets and model weights (after training).


## Files

* `DrawPHDimModelWeights.py`: (Not completed) Draw the results from `PHDimModelWeights`.

* `DrawPHDimPointCloud.py`: Draw results from `PHDimReport -> PointClouds`.

* `PHDimDisk.py`: Run experiments on point clouds, by adding noise from normal distribution `(0,eps_over_r)`, where
  `eps_over_r` is a list.
  
* `PHDimModelWeights.py`: Run experiments to calculate PH Dim from model weights taken from a folders (not pushed, too large) 
and PH Dim of the perturbed model weights.
  
* `PHDimPointCloud.py`: Calculate PH Dim from point clouds. The algorithm is adopted from 
  [Birdal et al. 2021](https://arxiv.org/pdf/2111.13171.pdf), [Adams et al. 2019](https://arxiv.org/pdf/1808.01079.pdf),
  [Jaquette et al. 2019](https://arxiv.org/pdf/1907.11182.pdf).
  
* `PHDimPointCloudNoise.py`: Run experiments to calculate PH Dim from point clouds with some kinds of (meaningless) perturbation.

* `TestSets.py`: Various sets from point clouds and some kinds of (meaningless) perturbation.

* `TrainModels.py`: Train DL models, code is adopted from [Birdal et al. 2021](https://arxiv.org/pdf/2111.13171.pdf) with
modifications. Usage
  ```
  python TrainModels.py --model <model_name: FC|AlexNet> --optimizer <optimizer: SGD|Adam> --learning_rate <default: 0.1>
  [--max_iter <default:10000> --batch_size <default:100> --dataset <mnist|cifar10|cifar100, default:mnist> --path <path_to_dataset>
  --eval_every <default:1000>]
  [[For FC] --width <default:100> --depth <default:3>]
  ```

* `utils.py`: Obsolete, subject to be deleted.
## Directories

### Models
Containing some deep learning models built by Pytorch: AlexNet, Fully Connected (tested), and LeNet (untested).
For the Fully Connected model, one has the option to specify model's width (number of neurons of each layer) and model's
depth (number of hidden layers + input and output layers).

### Results
Containing the results of some experiments. Divided into 2 directories.

* `PHDimReport`: The results in `.txt` form of the experiments. This directory is branched into other 3
sub-directory.
  
  * `PHDimModel`: The results of PH Dim of model weights. Calculated for 2 type of DL models: AlexNet and FC.
    * For `AlexNet.txt` - files without `avg` in name - the columns represent, respectively, `learning_rate`, `name_dataset`, `batch_size`, `optimizer_name`,
      `train_accuracy`, `test_accuracy`, `dimension_of_PH`, `alpha`, `estimated_PH_dim`.
      
    * For `AlexNet.txt`, - files with `avg` in name-  the columns represen  respectively, `learning_rate`, `name_dataset`, `batch_size`, `optimizer_name`,
      `train_accuracy`, `test_accuracy`, `dimension_of_PH`, `alpha`, `average_estimated_PH_dim`, `variance_PH_dim`.
      
    * For `FC.txt` the columns represent, respectively, `model_width`, `model_depth`,`learning_rate`, `name_dataset`, `batch_size`, `optimizer_name`,
      `train_accuracy`, `test_accuracy`, `dimension_of_PH`, `alpha`, `estimated_PH_dim`.
      
  * `PHDimModelWeights`: The supposedly directory containing the PHDim of model's weights after some kind of perturbations.
    It is devided into different sub-directories according to the type of perturbation: `FlipPoints`, `InversePoint`, 
    `Perturbed` (adding some values to some chosen points), `Reduced0` (reduce some points to 0),
    `SampleNoise` (adding some random noise from normal distribution to the points). The content of each file in these 
    sub-directories are organized as following: `model_info` (`model_name`|`dataset_name`|`batch_size`|`optimizer`|`learning_rate`)
    `condition of the points` (`True` if there is no noise, otherwise `Mutated`), `number_of_points`, `percentage_of_infected_points`,
    `dimension_of_PH`, `estimated_PH_dim`.
      
  * `PointClouds`: The results of PH Dim of points clouds. Currently, there are 2 types of sub-directories, ones start with
    `PHDimData` and not.
    
    * `PHDimData`: The directories containing the PH Dim of point clouds after some kind of perturbations. The type of perturbation
    is as in `PHDimModelWeights`. Each file in each of these folders correspond to a kind of (smooth) manifold in some dimension
      and is organized as follows:  `condition of the points` (`True` if there is no noise, otherwise `Mutated`), `number_of_points`, `percentage_of_infected_points`,
    `dimension_of_PH`, `estimated_PH_dim`.
        
    * `PHDimxDDisk`: The directories not start with `PHDimData`, containing the PH Dim of point clouds lie on the surface of a sphere
      after adding some noise taking from a normal distribution with respect to some parameters. For `2D`, `3D`, `10D` disks, these directories are divided into
      sub-directories, corresponding to the different radii of the spheres. Each of these sub-directories contains results from experiments. 
      The result files are organized as follows: `no_of_points`, `percent_infected_points`, `eps_over_r`, `dimension_of_PH`, `estimated_PH_dim`,  
      where `eps_over_r` is the parameter of normal distribution of noise, this can be taken w.r.t. the radius of the sphere
      or as fixed values. Due to different set-ups and computing limitations, the values of `eps_over_r` are taken as follows
      
        * Run 1 to Run 5: `radius = 5`, `eps_over_r = [0.1, 0.5, 0.8, 1, 1.2, 1.4, 1.5, 1.8, 2]`.
    
        * Run 6 to Run 10: `radius = 1`, `eps_over_r = np.arange(1, 2 * radius, 0.2)/radius`
    
        * Run 11 to Run 15: `radius = 2.5`, `eps_over_r = np.arange(1, 2 * radius, 0.2)/radius`
      
    * `PHDisk`: The last directory contains results of PH Dim of point clouds lie on the surface of a sphere
      after adding some noise taking from a normal distribution with respect to the dimension. For each dimension, 
      the files are organized as follows: `no_of_points`, `radius`, `percent_infected_points`, `eps_over_r`, `dimension_of_PH`, `estimated_PH_dim`,
      where `eps_over_r` is taken from the list `[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]`
      
* `Plots`: Containing some visualizations of the results from `PHDimReport`, mostly on point clouds. The name of the 
sub-directories is according to the above conventions of `PHDimReport` and self-explanatory. Note that some titles of the
  plot are mismatched. 
  * For the plots in `Models`, there are plots to show:
    * Correlation of generalization error (|test accuracy - train accuracy|) and PH dim (the `GenErrorPHDim` files), the 
      results are calculated by computing PH dim 5 or 10 times, taking the average and variance, sorting the generalization error 
      in increasing order and plot accordingly.
    * Correlation between training accuracy, testing accuracy, and PH Dim (the `TrainTestPHDim` files), the 
      results are calculated by computing PH dim 5 or 10 times, taking the average and variance, plot accordingly.
  

  