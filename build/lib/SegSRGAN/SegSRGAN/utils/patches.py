"""
  This software is governed by the CeCILL-B license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL-B
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".
  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited
  liability.
  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.
  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL-B license and that you accept its terms.
"""

import numpy as np
from itertools import product
from sklearn.feature_extraction.image import extract_patches
import SimpleITK as sitk
import scipy.ndimage
import sys
sys.path.insert(0, './utils')
from utils3d import modcrop3D
import os
import shutil
import time
import logging

def array_to_patches(arr, patch_shape=(3,3,3), extraction_step=1, normalization=False):
  #Make use of skleanr function extract_patches
  #https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/feature_extraction/image.py
  """Extracts patches of any n-dimensional array in place using strides.
  Given an n-dimensional array it will return a 2n-dimensional array with
  the first n dimensions indexing patch position and the last n indexing
  the patch content. 
  Parameters
  ----------
  arr : 3darray
      3-dimensional array of which patches are to be extracted
  patch_shape : integer or tuple of length arr.ndim
      Indicates the shape of the patches to be extracted. If an
      integer is given, the shape will be a hypercube of
      sidelength given by its value.
  extraction_step : integer or tuple of length arr.ndim
      Indicates step size at which extraction shall be performed.
      If integer is given, then the step is uniform in all dimensions.
  Returns
  -------
  patches : strided ndarray
      2n-dimensional array indexing patches on first n dimensions and
      containing patches on the last n dimensions. These dimensions
      are fake, but this way no data is copied. A simple reshape invokes
      a copying operation to obtain a list of patches:
      result.reshape([-1] + list(patch_shape))
  """
  
  patches = extract_patches(arr, patch_shape, extraction_step)    
  patches = patches.reshape(-1, patch_shape[0],patch_shape[1],patch_shape[2])    
  # patches = patches.reshape(patches.shape[0], -1) 
  if normalization==True:   
    patches -= np.mean(patches, axis=0)
    patches /= np.std(patches, axis=0)
  print('%.2d patches have been extracted' % patches.shape[0])  ,
  return patches

def patches_to_array(patches, array_shape, patch_shape=(3,3,3) ):
  #Adapted from 2D reconstruction from sklearn
  #https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/feature_extraction/image.py
  # SyntaxError: non-default argument follows default argument : exchange "array_shape" and "patch_shape"
  patches = patches.reshape(len(patches),*patch_shape)
  i_x, i_y, i_z = array_shape
  p_x, p_y, p_z = patch_shape
  array = np.zeros(array_shape)
  # compute the dimensions of the patches array
  n_x = i_x - p_x + 1
  n_y = i_y - p_y + 1
  n_z = i_z - p_z + 1
  for p, (i, j, k) in zip(patches, product(range(n_x), range(n_y), range(n_z))):
      array[i:i + p_x, j:j + p_y, k:k + p_z] += p
  
  for (i, j, k) in product(range(i_x), range(i_y), range(i_z)):
      array[i, j, k] /= float(min(i + 1, p_x, i_x - i) * min(j + 1, p_y, i_y - j) * min(k + 1, p_z, i_z - k))
  return array    

def create_patch_from_df_HR(df,
                            per_cent_val_max,
                            path_save_npy,
                            batch_size,
                            contrast_list,
                            list_res,
                            order=3,
                            normalisation=False,
                            thresholdvalue=0,
                            PatchSize=64,
                            stride=20,
                            is_conditional=False):
    
    Datas_list = []
    
    Labels_list = []
    
    Path_Datas_mini_batch=[]
    
    Path_Labels_mini_batch=[]
    

      
    try:
        
        os.makedirs(path_save_npy)
        print("Dossier cree")
    
    except OSError:
        
        logging.error('Unexpected error: the directory named %s already exists', path_save_npy)
    
    
    mini_batch=0
    
    remaining_patch=0
    
    for  i in range(df.shape[0]):
        
        ReferenceName = df["HR_image"].iloc[i] #path HR
        
        LabelName = df["Label_image"].iloc[i] #path label
        
        print('================================================================')
        print('Processing image : ', ReferenceName)
        
        t1=time.time()
        
        LowResolutionImage , ReferenceImage, LabelImage,UpScale = create_LR_HR_Label(ReferenceName,LabelName,list_res[i])
        
        border_to_keep = border_im_keep(ReferenceImage,thresholdvalue)
        
        ReferenceImage , LowResolutionImage = change_contrast(ReferenceImage,LowResolutionImage,contrast_list[i])
        
        LowResolutionImage = add_noise(LowResolutionImage,per_cent_val_max)
    
        InterpolatedImage , ReferenceImage = norm_and_interp(ReferenceImage,LowResolutionImage,order,UpScale)
        
        LabelImage , ReferenceImage , InterpolatedImage = remove_border(LabelImage,ReferenceImage,InterpolatedImage,border_to_keep) 
        
#        print LabelImage.shape
#        print ReferenceImage.shape
#        print InterpolatedImage.shape
#        print PatchSize
#        print stride
        
        HDF5Labels , HDF5Datas = create_patches(LabelImage,ReferenceImage,InterpolatedImage,PatchSize,stride,normalisation=False)
        
        np.random.seed(0)       # makes the random numbers predictable
        RandomOrder = np.random.permutation(HDF5Datas.shape[0])
        HDF5Datas = HDF5Datas[RandomOrder,:,:,:,:]
        HDF5Labels = HDF5Labels[RandomOrder,:,:,:,:]
        
        if is_conditional : 
            
            #list_res[i] = (res_x,res_y,res_z)
            HDF5Datas = np.concatenate((HDF5Datas,list_res[i][2]*np.ones_like(HDF5Datas)),axis=1) #1st axis = patch axis
        
        Datas_list.append(HDF5Datas)
        
        Labels_list.append(HDF5Labels)
        
        # Tranfer to array
        Datas = np.concatenate(np.asarray(Datas_list))
        Datas = Datas.reshape(-1,
                              Datas.shape[-4],
                              Datas.shape[-3],
                              Datas.shape[-2],
                              Datas.shape[-1]) 
        
        Labels = np.concatenate(np.asarray(Labels_list))
        Labels = Labels.reshape(-1,
                                Labels.shape[-4],
                                Labels.shape[-3],
                                Labels.shape[-2],
                                Labels.shape[-1]) 
                                
        t2=time.time()
        print("Image tranformation + patch creation and organisation :"+str(t2-t1))
                                
                                
        while Datas.shape[0]>=batch_size:   
            
            t1=time.time()
            
            np.save(path_save_npy+"/Datas_mini_batch_"+str(mini_batch)+".npy",Datas[:batch_size,:,:,:,:])
            
            t2=time.time()
            
            print("saving Data array :"+str(t2-t1))
            Datas = Datas[batch_size:,:,:,:,:]
            Datas_list = [Datas]
            
            t1=time.time()
            
            np.save(path_save_npy+"/Label_mini_batch_"+str(mini_batch)+".npy",Labels[:batch_size,:,:,:,:])
            
            t2=time.time()
            
            print("saving Label array :"+str(t2-t1))
            
            Labels = Labels[batch_size:,:,:,:,:]
            Labels_list = [Labels]
            
            
            Path_Datas_mini_batch.append(path_save_npy+"/Datas_mini_batch_"+str(mini_batch)+".npy")
            
            Path_Labels_mini_batch.append(path_save_npy+"/Label_mini_batch_"+str(mini_batch)+".npy")
            
            remaining_patch = Datas.shape[0]
            
            mini_batch+=1
                            
    return path_save_npy,Path_Datas_mini_batch , Path_Labels_mini_batch, remaining_patch  #Label[:,1,:,:,:] : seg , Label[:,0,:,:,:] : HR

def create_patches(Label,HR,interp,PatchSize,stride,normalisation=False):
    
        # Extract 3D patches
        print('Generating training patches ')
        
         
        DataPatch = array_to_patches(interp, 
                                     patch_shape=(PatchSize,PatchSize,PatchSize), 
                                     extraction_step = stride , 
                                     normalization=False) # image interp dim = (nb_patch,patch_size,patch_size,patch_size)
        print('for the interpolated low-resolution patches of training phase.')
                               
        
        LabelHRPatch = array_to_patches(HR, 
                                        patch_shape=(PatchSize,PatchSize,PatchSize), 
                                        extraction_step = stride , 
                                        normalization=False)# image HR dim = (nb_patch,patch_size,patch_size,patch_size)
                                        
        print('for the reference high-resolution patches of training phase.')
                                            
        
        LabelCortexPatch = array_to_patches(Label, 
                                            patch_shape=(PatchSize,PatchSize,PatchSize), 
                                            extraction_step = stride , 
                                            normalization=False) # image seg dim = (nb_patch,patch_size,patch_size,patch_size)
                                            
        print('for the Cortex Labels patches of training phase.') 
        
        # n-dimensional Caffe supports data's form : [numberOfBatches,channels,heigh,width,depth]         
        # Add channel axis !  
        HDF5Datas = DataPatch[:,np.newaxis,:,:,:] #ajoute une dimension de taille 1 en plus 
                
        # Concatenate HR patches and Cortex segmentation : HR patches in the 1st channel and Segmentation the in 2nd channel
        HDF5Labels = np.stack((LabelHRPatch,LabelCortexPatch)) #HDF5Labels[0] = LabelHRPatch et 1=LabelCortexPatch
        
        HDF5Labels = np.swapaxes(HDF5Labels,0,1) #premiere dim = patch ex : HDF5Labels[0,0,:,:,:] = HR premier patch et HDF5Labels[0,1,:,:,:] = Label premier patch
            
        
        return HDF5Labels,HDF5Datas
        
                  
                  
def border_im_keep(HR,thresholdvalue):
    
        darkRegionBox = np.where(HR>thresholdvalue)   
        border = ((np.min(darkRegionBox[0]),np.max(darkRegionBox[0])),
                  (np.min(darkRegionBox[1]),np.max(darkRegionBox[1])),
                  (np.min(darkRegionBox[2]),np.max(darkRegionBox[2])))
                  
        return border
        
def remove_border(Label,HR,interp,border) :


    HR = HR[border[0][0]:border[0][1],border[1][0]:border[1][1],border[2][0]:border[2][1]] 
    Label = Label[border[0][0]:border[0][1],border[1][0]:border[1][1],border[2][0]:border[2][1]]  
    interp = interp[border[0][0]:border[0][1],border[1][0]:border[1][1],border[2][0]:border[2][1]] 
       
    return Label,HR,interp
        
        
def norm_and_interp(HR,LR,order,UpScale):
    # Normalization by the max valeur of LR image
    MaxValue = np.max(LR)
    NormalizedReferenceImage =  HR/MaxValue
    NormalizedLowResolutionImage =  LR/MaxValue
        
    # Cubic Interpolation     
    InterpolatedImage = scipy.ndimage.zoom(NormalizedLowResolutionImage, 
                                  zoom = UpScale,
                                  order = order)
    
    return InterpolatedImage,NormalizedReferenceImage
        
def add_noise(LR,per_cent_val_max):
    
    sigma = per_cent_val_max*np.max(LR)

    LR = LR + np.random.normal(scale=sigma,size=LR.shape)
    
    LR[LR<0]=0

    return LR 

def create_LR_HR_Label(ReferenceName,LabelName,NewResolution):
        
        
        # Read NIFTI
        ReferenceNifti = sitk.ReadImage(ReferenceName)
        
        # Get data from NIFTI
        ReferenceImage = np.swapaxes(sitk.GetArrayFromImage(ReferenceNifti),0,2).astype('float32')
        
                # Read NIFTI
        LabelNifti = sitk.ReadImage(LabelName)
        
        # Get data from NIFTI
        LabelImage = np.swapaxes(sitk.GetArrayFromImage(LabelNifti),0,2).astype('float32')
        
        constant = 2*np.sqrt(2*np.log(2)) # As Greenspan et al. (Full_width_at_half_maximum : slice thickness)
        SigmaBlur = NewResolution/constant  
        
        
        # Get resolution to scaling factor
        UpScale = tuple(itemb/itema for itema,itemb in zip(ReferenceNifti.GetSpacing(),NewResolution))

        # Modcrop to scale factor
        ReferenceImage = modcrop3D(ReferenceImage,UpScale)
        
        LabelImage = modcrop3D(LabelImage,UpScale)
        

        # ===== Generate input LR image =====
        # Blurring
        BlurReferenceImage = scipy.ndimage.filters.gaussian_filter(ReferenceImage,
                                                            sigma = SigmaBlur)
                                                            
        print('Generating LR images with the resolution of ', NewResolution)
      
        # Downsampling
        LowResolutionImage = scipy.ndimage.zoom(BlurReferenceImage,
                                  zoom = (1/float(idxScale) for idxScale in UpScale),
                                  order = 0) 
                                  
        return LowResolutionImage,ReferenceImage,LabelImage,UpScale
        
def change_contrast(HR,LR,power):
    
    HR = HR**power
    LR = LR**power
    
    return HR , LR

        
def create_patch_from_image(ReferenceName,NewResolution,CortexName,PatchSize,batch_size,hdf5name,OutFile,order,thresholdvalue,stride):
    
        constant = 2*np.sqrt(2*np.log(2)) # As Greenspan et al. (Full_width_at_half_maximum : slice thickness)
        SigmaBlur = NewResolution/constant  
        
        print('================================================================')
        print('Processing image : ', ReferenceName)
        # Read NIFTI
        ReferenceNifti = sitk.ReadImage(ReferenceName)
        
        # Get data from NIFTI
        ReferenceImage = np.swapaxes(sitk.GetArrayFromImage(ReferenceNifti),0,2).astype('float32')
        
        # Get resolution to scaling factor
        UpScale = tuple(itemb/itema for itema,itemb in zip(ReferenceNifti.GetSpacing(),NewResolution))

        # Modcrop to scale factor
        ReferenceImage = modcrop3D(ReferenceImage,UpScale)

        # ===== Generate input LR image =====
        # Blurring
        BlurReferenceImage = scipy.ndimage.filters.gaussian_filter(ReferenceImage,
                                                            sigma = SigmaBlur)
                                                            
        print('Generating LR images with the resolution of ', NewResolution)
      
        # Downsampling
        LowResolutionImage = scipy.ndimage.zoom(BlurReferenceImage,
                                  zoom = (1/float(idxScale) for idxScale in UpScale),
                                  order = 0)  
        
        # Normalization by the max valeur of LR image
        MaxValue = np.max(LowResolutionImage)
        NormalizedReferenceImage =  ReferenceImage/MaxValue
        NormalizedLowResolutionImage =  LowResolutionImage/MaxValue
        
        # Cubic Interpolation     
        InterpolatedImage = scipy.ndimage.zoom(NormalizedLowResolutionImage, 
                                  zoom = UpScale,
                                  order = order)  
        
        # Processing cortex segmentation

        print('Processing cortex segmentation map : ', CortexName)
        CortexSegNifti = sitk.ReadImage(CortexName)
        CortexSegmentation = np.swapaxes(sitk.GetArrayFromImage(CortexSegNifti),0,2).astype('float32')
        
        # Shave region outside
        print('Remove the region outside the brain with the value of ', thresholdvalue)
        darkRegionBox = np.where(ReferenceImage>thresholdvalue)   
        border = ((np.min(darkRegionBox[0]),np.max(darkRegionBox[0])),
                  (np.min(darkRegionBox[1]),np.max(darkRegionBox[1])),
                  (np.min(darkRegionBox[2]),np.max(darkRegionBox[2])))     
        LabelHRImage = NormalizedReferenceImage[border[0][0]:border[0][1],border[1][0]:border[1][1],border[2][0]:border[2][1]] 
        LabelCortexImage = CortexSegmentation[border[0][0]:border[0][1],border[1][0]:border[1][1],border[2][0]:border[2][1]]  
        DataImage = InterpolatedImage[border[0][0]:border[0][1],border[1][0]:border[1][1],border[2][0]:border[2][1]]   
        
        # Extract 3D patches
        print('Generating training patches with the resolution of ', NewResolution, ' : ')
        DataPatch = array_to_patches(DataImage, 
                                     patch_shape=(PatchSize,PatchSize,PatchSize), 
                                     extraction_step = stride , 
                                     normalization=False) # image interp dim = (nb_patch,patch_size,patch_size,patch_size)
        print('for the interpolated low-resolution patches of training phase.')                                 
        LabelHRPatch = array_to_patches(LabelHRImage, 
                                        patch_shape=(PatchSize,PatchSize,PatchSize), 
                                        extraction_step = stride , 
                                        normalization=False)# image HR dim = (nb_patch,patch_size,patch_size,patch_size)
        print('for the reference high-resolution patches of training phase.')        
        LabelCortexPatch = array_to_patches(LabelCortexImage, 
                                            patch_shape=(PatchSize,PatchSize,PatchSize), 
                                            extraction_step = stride , 
                                            normalization=False) # image seg dim = (nb_patch,patch_size,patch_size,patch_size)
        print('for the cortex segmentation patches of training phase.')
                          
        # n-dimensional Caffe supports data's form : [numberOfBatches,channels,heigh,width,depth]         
        # Add channel axis !  
        HDF5Datas = DataPatch[:,np.newaxis,:,:,:] #ajoute une dimension de taille 1 en plus 
                
        # Concatenate HR patches and Cortex segmentation : HR patches in the 1st channel and Segmentation the in 2nd channel
        HDF5Labels = np.stack((LabelHRPatch,LabelCortexPatch)) #HDF5Labels[0] = LabelHRPatch et 1=LabelCortexPatch
        HDF5Labels = np.swapaxes(HDF5Labels,0,1) #premiere dim = patch ex : HDF5Labels[0,0,:,:,:] = HR premier patch et HDF5Labels[0,1,:,:,:] = Label premier patch
            
        # Rearrange
        np.random.seed(0)       # makes the random numbers predictable
        RandomOrder = np.random.permutation(HDF5Datas.shape[0])
        HDF5Datas = HDF5Datas[RandomOrder,:,:,:,:]
        HDF5Labels = HDF5Labels[RandomOrder,:,:,:,:]
                   

    
    
    
    
    
