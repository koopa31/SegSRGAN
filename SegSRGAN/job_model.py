# -*- coding: utf-8 -*-
import Function_for_application_test_python3
import glob
import os
import argparse
import ast
import sys
import requests

import numpy as np
import pandas as pd

from SegSRGAN.utils.download import download_weights

def absolute_weights_path(path):
    """
    Turn the weights path to its absolute path if relative.
    :param path: Path to the weights directory
    :return: Absolute path to the weights directory
    """
    if os.path.isabs(path) is False:
        weights_path = os.path.join(os.path.split(__file__)[0], path)
    else:
        weights_path = path
    return weights_path

# downloading of the weights in case of the use of the pip package
download_weights(absolute_weights_path('weights'))

# Fonction which will be used hereafter :


start = "\033[1m" # for printing in bold
end = "\033[0;0m"
RED = '\033[31m'   # mode 31 = red forground
RESET = '\033[0m'  # mode 0  = reset




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
        
        
def result_folder_name(base_folder,patch,step,result_folder) :
    
    base_folder_split =  os.path.split(base_folder)
    
    path_output = os.path.join(*base_folder_split[:(len(base_folder_split) - 1)],"Result_with_"+result_folder,"patch "+str(
        patch))

    if patch is None : 

        path_output_cortex = os.path.join(path_output,"Cortex whole image.nii.gz")

        path_output_SR = os.path.join(path_output,"SR  whole image.nii.gz")
        
    else : 
        
         path_output = path_output + " step " + str(step) 
         
         path_output_cortex = os.path.join(path_output,"Cortex patch " + str(patch) + " step " + str(step) + ".nii.gz")

         path_output_SR = os.path.join(path_output,"SR patch " + str(patch) + " step " + str(step) + ".nii.gz")
         
    return path_output, path_output_cortex, path_output_SR




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

def list_of_weights():
    """
    Gets the list of all the existing weights from the Github repository
    """
    z = requests.get('https://api.github.com/repos/koopa31/SegSRGAN/contents/data/weights?ref=master')
    contents = z.json()
    weights_list= []
    for content in contents:
        weights_list.append(os.path.join('weights', content['name']))
    return weights_list

weights_list = list_of_weights()


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, help="Path of the csv file")
parser.add_argument("-dp", "--debut_path", type=str, help="Path beginning of the csv (default: %(default)s voxels)",
                    default='')
parser.add_argument("-pa", "--patch", type=str, help="Patch size (default: %(default)s)", default=128)
parser.add_argument("-s", "--step", type=str, help="Step between patches. Must be a tuple of tuple (default: "
                                                   "%(default)s)", default=64)
parser.add_argument("-rf", "--result_folder_name", type=str, help='Name of the folder where the result is going to be '
                                                                  'stored')
parser.add_argument("-wp", "--weights_path", type=str, help='Weights relative path. List of the available weights:'
                                                            ' %(default)s', default=weights_list)
parser.add_argument("-bb", "--by_batch", type=str, help="Prediction on list of patches instead of using a for loop. "
                                                         "Enables for instance to automatically computes in multi-gpu "
                                                         "mode(default: %(default)s)", default="False")

parser.add_argument('-n', '--new_low_res', type=str, help='Resolution of results (SR and segmentation).' 
                    'Ex : 0.5,0.5,0.5 (default) ',default='0.5,0.5,0.5')


args = parser.parse_args()

resolution = args.new_low_res



by_batch = ast.literal_eval(args.by_batch)

# Argument :
# name of the result folder
result_folder = args.result_folder_name

weights_path = args.weights_path

debut_relatif_path = args.debut_path  # Path to Base_pour_romeo

data = pd.read_csv(os.path.join(debut_relatif_path, args.path), header=None).iloc[:, 0].sort_values()


path_pour_application = [os.path.join(debut_relatif_path, i) for i in data]


resolution = tuple(list_of(resolution,float))

if len(resolution) != 3 :
    
    raise AssertionError("\n"+RED+start+'The resolution have to be have size 3 !'+end+RESET+"\n")

ensemble_pas = list_of_lists(args.step)
patchs = list_of(args.patch,int)

if len(ensemble_pas) != len(patchs):
    
    raise AssertionError("\n"+RED+start+'Each patch size need to have its own step (the number of "," in the patch argument must to be the same as the one in the step argument)'+end+RESET+"\n")

         
# Execution :


ensemble_pas = pd.DataFrame(ensemble_pas, index=patchs)

np_pas_per_patch = ensemble_pas.shape[1]

weights_path = absolute_weights_path(weights_path)

for i in path_pour_application:

    for patch in patchs:

            for step in np.unique(ensemble_pas.loc[patch]):
                
                path_output, path_output_cortex, path_output_SR = result_folder_name(i,patch,step,result_folder)
        
                print("\n"+start+"Processing : "+path_output+"\n"+end)

                if not os.path.exists(path_output):

                    create_folder(path_output)
                
                if len(os.listdir(path_output))==0 :
                    
                    try : 
                        
                        Function_for_application_test_python3.segmentation(input_file_path=i,
                                                                           step=step,
                                                                           new_resolution=resolution,
                                                                           patch=patch,
                                                                           path_output_cortex=path_output_cortex,
                                                                           path_output_hr=path_output_SR,
                                                                           weights_path=weights_path,
                                                                           by_batch=by_batch,
                                                                           )
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
