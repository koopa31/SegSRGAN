#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import progressbar
import sys
import numpy as np
import SimpleITK as sitk
#sys.path.insert(0,os.path.split(__file__)[0])
from utils.utils3d import shave3D
from utils.utils3d import pad3D
from utils.ImageReader import NIFTIReader
from utils.ImageReader import DICOMReader
import pandas as pd
import argparse

import tensorflow as tf

from utils.u_net import UNet

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

GREEN = '\033[32m' # mode 32 = green forground
start = "\033[1m" # for printing in bold
end = "\033[0;0m"
RESET = '\033[0m'  # mode 0  = reset
RED = '\033[31m'   # mode 31 = red forground


class u_net_test(object):

    def __init__(self, weights,patch1,patch2,patch3):
        
         self.net = UNet((1,patch1,patch2,patch3), out_ch=1, start_ch=28, depth=2, inc_rate=2., activation='relu', 
 		 dropout=0, batchnorm=True, maxpool=False, upconv=True, residual=False)
         
         self.net.load_weights(weights)
         
         self.patch1,self.patch2,self.patch3 = patch1,patch2,patch3

    def get_patch(self):
        """

        :return:
        """
        return self.patch

    def test_by_patch(self, test_image, step=1):
        """

        :param test_image: Image to be tested
        :param step: step
        :param by_batch: to enable by batch processing
        :return:
        """
        # Init temp
        height, width, depth = np.shape(test_image)
        temp_seg = np.zeros_like(test_image)
        weighted_image = np.zeros_like(test_image)


        i = 0
        bar = progressbar.ProgressBar(maxval=len(np.arange(0, height - self.patch1 + 1, step)) * len(
            np.arange(0, width - self.patch2 + 1, step)) * len(np.arange(0, depth - self.patch3 + 1, step))).\
            start()
        print('Patch=', self.patch1)
        print('Step=', step)
        for idx in range(0, height - self.patch1 + 1, step):
            for idy in range(0, width - self.patch2 + 1, step):
                for idz in range(0, depth - self.patch3 + 1, step):
                    # Cropping image
                    test_patch = test_image[idx:idx + self.patch1, idy:idy + self.patch2, idz:idz + self.patch3]
                    image_tensor = test_patch.reshape(1, 1, self.patch1, self.patch2, self.patch3).\
                        astype(np.float32)
                    predict_patch = self.net.predict([image_tensor], batch_size=1)

                    temp_seg[idx:idx + self.patch1, idy:idy + self.patch2, idz:idz + self.patch3] += \
                        predict_patch[0,0, :, :, :]
                        
                    weighted_image[idx:idx + self.patch1, idy:idy + self.patch2,
                    idz:idz + self.patch3] += np.ones_like(predict_patch[0, 0, :, :, :])

                    i += 1

                    bar.update(i)
                        
        estimated_segmentation = temp_seg / weighted_image
  
        return estimated_segmentation


def segmentation(input_file_path, step, path_output_cortex, weights_path,patch=None):


    # Read low-resolution image
    if input_file_path.endswith('.nii.gz') or input_file_path.endswith('.hdr'):
        image_instance = NIFTIReader(input_file_path)
    elif os.path.isdir(input_file_path):
        image_instance = DICOMReader(input_file_path)

    itk_image = image_instance.itk_image
    
    image_np = image_instance.get_np_array()

    if patch is not None:

        print("patch given")

        patch1 = patch2 = patch3 = int(patch)

        border = (
        int((image_np.shape[0] - int(patch)) % step), int((image_np.shape[1] - int(patch)) % step),
        int((image_np.shape[2] - int(patch)) % step))

        border_to_add = (step - border[0], step - border[1], step - border[2])

        # padd border
        padded_image_np = pad3D(image_np, border_to_add)  # remove border of the image

    else:
        border = (
        int(image_np.shape[0] % 4), int(image_np.shape[1] % 4), int(image_np.shape[2] %
                                                                                        4))
        border_to_add = (4 - border[0], 4 - border[1], 4 - border[2])

        padded_image_np = pad3D(image_np, border_to_add)   # remove border of the image

        height, width, depth = np.shape(padded_image_np)
        patch1 = height
        patch2 = width
        patch3 = depth

    if ((step>patch1) |  (step>patch2) | (step>patch3)) & (patch is not None) :

        raise AssertionError('The step need to be smaller than the patch size')

    if (np.shape(padded_image_np)[0]<patch1)|(np.shape(padded_image_np)[1]<patch2)|(np.shape(padded_image_np)[2]<patch3):

        raise AssertionError('The patch size need to be smaller than the interpolated image size')

    # Loading weights
    u_net_test_instance = u_net_test(weights_path, patch1, patch2, patch3)

    # GAN
    print("Testing : ")
    estimated_cortex = u_net_test_instance.test_by_patch(padded_image_np, step=step)
    # parcours de l'image avec le patch

    # Padding
    # on fait l'operation de padding a l'envers
    estimated_cortex = shave3D(estimated_cortex, border_to_add)

    # Cortex segmentation
    output_cortex = sitk.GetImageFromArray(np.swapaxes(estimated_cortex, 0, 2))
    output_cortex.SetSpacing(tuple(np.array(image_instance.itk_image.GetSpacing())))
    output_cortex.SetOrigin(itk_image.GetOrigin())
    output_cortex.SetDirection(itk_image.GetDirection())

    sitk.WriteImage(output_cortex, path_output_cortex)

    return "Segmentation Done"




def create_folder(directory):
    """
    Creates a folder if it does not already exist
    :param directory: path of the directory
    :return:
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print(RED+start+'Error: Creating directory. ' + directory+end+RESET)
        
        
def result_folder_name(base_folder, patch, step, result_folder) :
    
    base_folder_split = os.path.split(base_folder)
    
    path_output = os.path.join(*base_folder_split[:(len(base_folder_split) - 1)], "Result_with_" + result_folder,
                               "patch_" + str(patch))

    if patch is None:

        path_output_cortex = os.path.join(path_output, "Cortex_whole_image.nii.gz")
  
    else:
        
         path_output = path_output + "_step_" + str(step)
         
         path_output_cortex = os.path.join(path_output, "Cortex_patch_" + str(patch) + "_step_" + str(step) + ".nii.gz")

    return path_output, path_output_cortex




def pgcd(a,b):
    #Retourne le PGCD de a et b  
    r=1
    if a<b:
        a,b=b,a
    if a!=0 and b!=0:
        while r > 0:
            r=a%b
            a=b
            b=r
        return(a)
        
def ppcm_function(a,b):
    
    #Retourne le PPCM de a et b
    if a!=0 and b!=0:
        ppcm = (a*b)/pgcd(a,b)
        return(int(ppcm))


def list_of_lists(arg):
    
    m = [x.split(' ') for x in arg.split(',')]
    size_of_step_per_patch = [ len(x) for x in m]

    if len(m) > 1 :
        
        ppcm = size_of_step_per_patch[0]
        
        for i in range(1,len(size_of_step_per_patch)) : 
            
            ppcm = ppcm_function(ppcm,size_of_step_per_patch[i])
            
        for i in range(len(m)) : 
            
            mult = ppcm / size_of_step_per_patch[i]
            
            m[i] = list(np.repeat(m[i],mult))
            
    m = np.array(m).astype('int')
    
    return m.tolist()


def list_of(arg,result_type=int):
    
    m=[]
    for x in arg.split(',') : 
        
        if x != "None":
            m.append(result_type(x))
        else : 
            m.append(None)
        
    return m


parser = argparse.ArgumentParser()

parser.add_argument("-p", "--path", type=str, help="Path of the csv file")
parser.add_argument("-pa", "--patch", type=str, help="Patch size (default: %(default)s)", default=128)
parser.add_argument("-s", "--step", type=str, help="Step between patches. Must be a tuple of tuple (default: "
                                                   "%(default)s)", default=64)
parser.add_argument("-rf", "--result_folder_name", type=str, help='Name of the folder where the result is going to be '
                                                                  'stored')
parser.add_argument("-wp", "--weights_path", type=str, help='Weights relative path. List of the available weights:'
                                                            ' %(default)s')

args = parser.parse_args()


# Argument :
# name of the result folder
result_folder = args.result_folder_name

weights_path = args.weights_path


data = pd.read_csv(args.path, header=None).iloc[:, 0].sort_values()


path_pour_application = data.tolist()


ensemble_pas = list_of_lists(args.step)
patchs = list_of(args.patch,int)

if len(ensemble_pas) != len(patchs):
    
    raise AssertionError("\n"+RED+start+'Each patch size need to have its own step (the number of "," in the patch argument must to be the same as the one in the step argument)'+end+RESET+"\n")

         
# Execution :


ensemble_pas = pd.DataFrame(ensemble_pas, index=patchs)

np_pas_per_patch = ensemble_pas.shape[1]

for i in path_pour_application:

    for patch in patchs:

            for step in np.unique(ensemble_pas.loc[patch]):
                
                path_output, path_output_cortex = result_folder_name(i,patch,step,result_folder)
        
                print("\n"+start+"Processing : "+path_output+"\n"+end)

                if not os.path.exists(path_output):

                    create_folder(path_output)
                
                if len(os.listdir(path_output))==0 :
                    
                    try : 
                    
                        segmentation(i,step, path_output_cortex, weights_path,patch)
                        
                    except (Exception,KeyboardInterrupt) as err :  # on attrape les erreur et on execute ce qu'il se passe en dessus (ici pour les erreur Exception et KeyboardInterrupt )
                        
                        if str(err) != "" :
                            
                            print('\n'+RED+start+'ERROR: %s' % str(err))
                            print("\n"+RESET+end)
                           
                        if len(os.listdir(path_output))==0 :
                            
                            print("\n"+start+"The folder ",path_output,"have been deleted ! \n"+end)
                            
                            os.rmdir(path_output) # Delete the folder path_output because in case of error the folder was already created but empty.
                            
                        else : 
                            
                            print("\n"+RED+start+"An error occurs but the result folder ",path_output," is not empty so cannot be deleted."+end+RESET+"\n")
                            
                        sys.exit()
                        
                    
                else:
                    
                    

                    print("\n"+start+"The output folder : ", path_output,"already exists and is not empty. Generally because the result have already been computed"+end,"\n")


