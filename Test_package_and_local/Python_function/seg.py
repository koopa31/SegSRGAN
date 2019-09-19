import os
import sys
from pathlib import Path
import importlib
import argparse

"""parent=Path(__file__).resolve().parent
parent_parent=parent.parent

sys.path.insert(0,os.path.join(str(parent),".."))

print(Path(__file__))
print(Path(__file__).resolve())
print(parent_parent)
print(parent)
print(os.path.join(str(parent),".."))"""

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image_path", type=str, help="Path of the image")

args = parser.parse_args()


from SegSRGAN.Function_for_application_test_python3 import segmentation


wpath = "weights/Perso_without_data_augmentation"


# Find the weight location downloaded via the module import.
wpath_base=importlib.util.find_spec("SegSRGAN").submodule_search_locations[0]
wpath=os.path.join( wpath_base, wpath)

# Path to nifti file (TO BE ADAPTED).
input_nii = args.image_path
output_cortex_nii = os.path.join(str(Path(input_nii).resolve().parent),"Result_segmentation_function","cortex.nii.gz")
output_hr_nii = os.path.join(str(Path(input_nii).resolve().parent),"Result_segmentation_function","SR.nii.gz") 

print(output_cortex_nii)

# Segmentation.
segmentation( input_file_path = input_nii,
              step = 128,
              new_resolution = (0.5,0.5,0.5),
              patch=128,
              path_output_cortex = output_cortex_nii,
              path_output_hr = output_hr_nii,
              weights_path=wpath )


                        
