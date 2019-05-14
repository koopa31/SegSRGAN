# SegSRGAN

## Installation

`pip install SegSRGAN`

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
> * **is_conditional** to perform a conditional GAN on the LR image resolution


## Segmentation of a set of images with several step and patch values

In order to facilitate the segmentation of several images, you can run SegSRGAN/SegSRGAN/job_model.py:

`python job_model.py --path
--patch --step --result_folder_name --weights_relative_path --is_conditional`

The list of the paths of the images to be processed must be stored in a csv file.

Where:

> * **path** Path of the csv file
> * **patch** list of Patch sizes (example: 64 128)
> * **step** list of steps (example: 32 64,64 128 in this example we run steps 32 and 64 for 
    patch 64 and steps 64 and 128 for patch 128)
> * **result_folder_name** Name of the folder containing the results
> * **is_conditional** Boolean to perform a conditional neural network with a condition on z-resolution
