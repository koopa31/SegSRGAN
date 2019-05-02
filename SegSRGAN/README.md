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
