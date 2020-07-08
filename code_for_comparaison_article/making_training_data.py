import sys 
import os
import glob
import SimpleITK as sitk
import numpy as np
import logging
import time
import pandas as pd
import argparse

#sys.path.insert(0,os.path.split(os.path.split(__file__)[0])[0])

sys.path.insert(0,"/home/quentin/Romeo/Segmentation_neural_network/SegSRGAN/code_for_comparaison_article/")


import utils.patches as patch 
from utils.ImageReader import NIFTIReader
from utils.ImageReader import DICOMReader


def create_patches(SRReCNN3D_result_path_list,Label_path_list,HR_path_list,path_save_npy,thresholdvalue = 0,patch_size = 64,stride = 20,batch_size = 32): 
    mini_batch = 0
    
    data_list = []
    
    labels_list = []
    
    path_data_mini_batch = []
    
    path_labels_mini_batch = []
    
    try:
        os.makedirs(path_save_npy)
        print("Dossier cree")
    
    except OSError:
        logging.error('Unexpected error: the directory named %s already exists', path_save_npy)
    
    mini_batch = 0
    
    remaining_patch = 0
    
    for i in range(len(SRReCNN3D_result_path_list)) :
        
        print(SRReCNN3D_result_path_list[i])
        SRReCNN3D = NIFTIReader(SRReCNN3D_result_path_list[i])
        Label = NIFTIReader(Label_path_list[i])
        HR = NIFTIReader(HR_path_list[i])
        
        np_SRReCNN3D = SRReCNN3D.get_np_array()
        np_Label = Label.get_np_array()
        np_HR = HR.get_np_array()
        
        
        np_Label = patch.put_label_and_sr_to_same_dimension(np_SRReCNN3D,np_Label) #Upscale can imply dimension reduction. ex : upscale of 6 factor can't result into 203 (not mutiple of 6) shaped vector. In this case we suppose the last value removed. 
        
        border = patch.border_im_keep(np_HR, thresholdvalue)
        
        np_SRReCNN3D , np_Label = patch.remove_border(np_SRReCNN3D,np_Label,border)
        
        hdf5_data,hdf5_labels = patch.create_patches(np_SRReCNN3D,np_Label,patch_size,stride)
        
        np.random.seed(0)       # makes the random numbers predictable
        random_order = np.random.permutation(hdf5_data.shape[0])
        hdf5_data = hdf5_data[random_order, :, :, :, :]
        hdf5_labels = hdf5_labels[random_order, :, :, :, :]
        
        data_list.append(hdf5_data)
        labels_list.append(hdf5_labels)
        
        # Transfer to array
        datas = np.concatenate(np.asarray(data_list))
        datas = datas.reshape(-1,
                          datas.shape[-4],
                          datas.shape[-3],
                          datas.shape[-2],
                          datas.shape[-1])
        
        labels = np.concatenate(np.asarray(labels_list))
        labels = labels.reshape(-1,
                            labels.shape[-4],
                            labels.shape[-3],
                            labels.shape[-2],
                            labels.shape[-1])
        
        while datas.shape[0] >= batch_size:
        
            t1 = time.time()
        
            np.save(os.path.join(path_save_npy ,"Datas_mini_batch_"+str(mini_batch)) + ".npy",
                    datas[:batch_size, :, :, :, :])
        
            t2 = time.time()
        
            print("saving Data array :"+str(t2-t1))
            datas = datas[batch_size:, :, :, :, :]
            data_list = [datas]
        
            t1 = time.time()
        
            np.save(os.path.join(path_save_npy,"Label_mini_batch_" + str(mini_batch) + ".npy"), 
                    labels[:batch_size, :, :, :, :])
        
            t2 = time.time()
        
            print("saving Label array :" + str(t2-t1))
        
            labels = labels[batch_size:, :, :, :, :]
            labels_list = [labels]
        
            path_data_mini_batch.append(os.path.join(path_save_npy,"Datas_mini_batch_" + str(mini_batch) + ".npy"))
        
            path_labels_mini_batch.append(os.path.join(path_save_npy,"Label_mini_batch_" + str(mini_batch) + ".npy"))
        
            remaining_patch = datas.shape[0]
        
            mini_batch += 1
            
    print(remaining_patch,"Patch will not be processed")
    
    return path_data_mini_batch,path_labels_mini_batch



parser = argparse.ArgumentParser()
parser.add_argument("-csv_path", "--csv_path", type=str, help="Path of the csv file")
parser.add_argument("-path_save_npy", "--path_save_npy", type=str, help="Path for saving training and testing file")
    
    
args = parser.parse_args()


data = pd.read_csv(args.csv_path)
    
SRReCNN3D_result_path_list_train =  data[data["Base"]=="Train"]["SR_path"].tolist()
Label_path_list_train = data[data["Base"]=="Train"]["Label_path"].tolist()
HR_path_list_train = data[data["Base"]=="Train"]["HR_path"].tolist()
path_save_npy_train = args.path_save_npy+"/Train"

SRReCNN3D_result_path_list_test =  data[data["Base"]=="Test"]["SR_path"].tolist()
Label_path_list_test = data[data["Base"]=="Test"]["Label_path"].tolist()
HR_path_list_test = data[data["Base"]=="Test"]["HR_path"].tolist()
path_save_npy_test = args.path_save_npy+"/Test"

print("making Training data : \n")
create_patches(SRReCNN3D_result_path_list_train,Label_path_list_train,HR_path_list_train,path_save_npy_train,thresholdvalue = 0,patch_size = 64,stride = 20,batch_size = 32)
print("making Testing data : \n")
create_patches(SRReCNN3D_result_path_list_test,Label_path_list_test,HR_path_list_test,path_save_npy_test,thresholdvalue = 0,patch_size = 64,stride = 20,batch_size = 1)

