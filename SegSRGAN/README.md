# SegSRGAN

This algorithm is based on the [method](https://hal.archives-ouvertes.fr/hal-01895163) proposed by Chi-Hieu Pham in 2019.

## Installation

`pip install SegSRGAN`

## Perform a training

`python SegSRGAN_train_avec_base_test.py --newlowres --csv  --snapshot_folder --dice_file --mse_file --folder_training_data`

> * **csv** file which contains the paths to the files for the training. They are divided into two categories: train and test. As a consequence, it must contain 3 columns respectively called: HR\_image, Label\_image and Base (which can be equal to either Train or Test).  
> * **dice_file** CSV file where to store the DICE at each epoch (_string_)
> * **mse\_file** MSE file where to store the DICE at each epoch (*string*)
> * **folder\_training\_data** folder which contains the training images database (*string*)
> * **epoch** number of training epochs (*integer*)
> * **batchsize** number of patches per mini batch (*integer*)
> * **snapshot\_folder** how often the weights are saved on the disk (for instance if equal to 2, the weights are saved on the disk one epoch in two)(*integer*)}
> * **numcritic** how many times we train the discriminator before training the generator (*integer*)


But it is also possible to continue a training from its saved weight, adding the following parameters: 

> * **initepoch** number of the epoch from which the training will continue (*integer*)
> * **weights** path to the saved weights from which the training will continue (*string*)


Two very important parameters to set the structure of the network are:

> * **kernelgen**  Number of output channel of the first convolutional layer of the generator (see \hyperref[architecture]{ section \ref*{architecture}})
> * **kerneldis** Number of output channel of the first convolutional layer of the discriminator

## Perform a segmentation

`from SegSRGAN.SegSRGAN.Function_for_application_test_python3 import segmentation`

`segmentation(input_file_path, step, NewResolution, path_output_cortex, path_output_HR, weights_path, patch=None,
                 spline_order=3, by_batch=False, is_conditional=False)`
                 
Where:
> * **input_file_path** is the path of the image to be super resolved and segmented 
> * **step** is the shifting step for the patches
> * **NewResolution** is the new z-resolution we want for the output image 
> * **path_output_cortex** output path of the segmented cortex
> * **path_output_HR** output path of the super resolution output image
> * **weights_path** is the path of the file which contains the pre-trained weights for the neural network
> * **patch** is the size of the patches
> * **spline_order** for the interpolation
> * **by_batch** is to enable the by-batch processing



## Segmentation of a set of images with several step and patch values

In order to facilitate the segmentation of several images, you can run SegSRGAN/SegSRGAN/job_model.py:

`python job_model.py --path
--patch --step --result_folder_name --weights_path`

The list of the paths of the images to be processed must be stored in a csv file.

Where:

> * **path** Path of the csv file
> * **patch** list of patch sizes 
> * **step** list of steps 
> * **result_folder_name** Name of the folder containing the results

Example of syntax for step and patch setting:

--patch 64,128

--step 32 64,64 128

In this example we run steps 32 and 64 for patch 64 and steps 64 and 128 for patch 128. Be careful to respect the exact same spaces.

