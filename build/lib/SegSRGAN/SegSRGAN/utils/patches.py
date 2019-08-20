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

import os
import shutil
import time
import logging
import sys
import scipy.ndimage

import numpy as np

from itertools import product
from sklearn.feature_extraction.image import extract_patches
from ImageReader import NIFTIReader
from ImageReader import DICOMReader
sys.path.insert(0, os.path.join('.','utils'))
from utils.utils3d import modcrop3D


import SimpleITK as sitk


def array_to_patches(arr, patch_shape=(3, 3, 3), extraction_step=1, normalization=False):
    # Make use of skleanr function extract_patches
    # https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/feature_extraction/image.py
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
    normalization : bool
        Enable normalization of the patches

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
    patches = patches.reshape(-1, patch_shape[0], patch_shape[1], patch_shape[2])
    # patches = patches.reshape(patches.shape[0], -1)
    if normalization is True:
        patches -= np.mean(patches, axis=0)
        patches /= np.std(patches, axis=0)
    print('%.2d patches have been extracted' % patches.shape[0])
    return patches


def patches_to_array(patches, array_shape, patch_shape=(3, 3, 3)):
    """
    Swicth from the patches to the image
    :param patches: patches array
    :param array_shape: shape of the array
    :param patch_shape: shape of the patches
    :return: array
    """
    # Adapted from 2D reconstruction from sklearn
    # https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/feature_extraction/image.py
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


def create_patch_from_df_hr(df,
                            per_cent_val_max,
                            path_save_npy,
                            batch_size,
                            contrast_list,
                            list_res,
                            patch_size,
                            order=3,
                            thresholdvalue=0,
                            stride=20,
                            is_conditional=False):
    
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
    
    for i in range(df.shape[0]):
        
        reference_name = df["HR_image"].iloc[i]
        # path HR
        
        label_name = df["Label_image"].iloc[i]
        # path label
        
        print('================================================================')
        print('Processing image : ', reference_name)
        
        t1 = time.time()
        
        low_resolution_image, reference_image, label_image, up_scale = create_lr_hr_label(reference_name, label_name,
                                                                                          list_res[i]) #From here, the three images have the same size (see crop in create_lr_hr_label)
        
        border_to_keep = border_im_keep(reference_image, thresholdvalue)
        
        reference_image, low_resolution_image = change_contrast(reference_image, low_resolution_image, contrast_list[i])
        
        low_resolution_image = add_noise(low_resolution_image, per_cent_val_max)
    
        interpolated_image, reference_image = norm_and_interp(reference_image, low_resolution_image, order, up_scale)
        
        label_image, reference_image, interpolated_image = remove_border(label_image, reference_image,
                                                                           interpolated_image, border_to_keep)
        
        if (patch_size>interpolated_image.shape[0])|(patch_size>interpolated_image.shape[1]) | (patch_size>interpolated_image.shape[2]) : 
            
            raise AssertionError('The patch size is too large compare to the size on the image')

        hdf5_labels, had5_dataa = create_patches(label_image, reference_image, interpolated_image, patch_size, stride)
        
        np.random.seed(0)       # makes the random numbers predictable
        random_order = np.random.permutation(had5_dataa.shape[0])
        had5_dataa = had5_dataa[random_order, :, :, :, :]
        hdf5_labels = hdf5_labels[random_order, :, :, :, :]
        
        if is_conditional:
            
            # list_res[i] = (res_x,res_y,res_z)
            # 1st axis = patch axis
            had5_dataa = np.concatenate((had5_dataa, list_res[i][2]*np.ones_like(had5_dataa)), axis=1)
        
        data_list.append(had5_dataa)
        
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
                                
        t2 = time.time()
        print("Image tranformation + patch creation and organisation :"+str(t2-t1))

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

    # Label[:,1,:,:,:] : seg , Label[:,0,:,:,:] : HR
    return path_save_npy, path_data_mini_batch, path_labels_mini_batch, remaining_patch


def create_patches(label, hr, interp, patch_size, stride):
    
    # Extract 3D patches
    print('Generating training patches ')

    data_patch = array_to_patches(interp, patch_shape=(patch_size, patch_size, patch_size), extraction_step=stride,
                                  normalization=False)
    # image interp dim = (nb_patch,patch_size,patch_size,patch_size)
    print('for the interpolated low-resolution patches of training phase.')

    label_hr_patch = array_to_patches(hr, patch_shape=(patch_size, patch_size, patch_size), extraction_step=stride,
                                      normalization=False)
    # image hr dim = (nb_patch,patch_size,patch_size,patch_size)

    print('for the reference high-resolution patches of training phase.')

    label_cortex_patch = array_to_patches(label, patch_shape=(patch_size, patch_size, patch_size),
                                          extraction_step=stride, normalization=False)
    # image seg dim = (nb_patch,patch_size,patch_size,patch_
    # size)

    print('for the Cortex Labels patches of training phase.')

    # n-dimensional Caffe supports data's form : [numberOfBatches,channels,heigh,width,depth]
    # Add channel axis !
    hdf5_data = data_patch[:, np.newaxis, :, :, :]
    # ajoute une dimension de taille 1 en plus

    # Concatenate hr patches and Cortex segmentation : hr patches in the 1st channel and Segmentation the in 2nd channel
    hdf5_labels = np.stack((label_hr_patch, label_cortex_patch))
    # hdf5_labels[0] = label_hr_patch et 1=label_cortex_patch

    hdf5_labels = np.swapaxes(hdf5_labels, 0, 1)
    # premiere dim = patch ex : hdf5_labels[0,0,:,:,:] = hr first patch
    # et hdf5_labels[0,1,:,:,:] = label first patch

    return hdf5_labels, hdf5_data

                  
def border_im_keep(hr, threshold_value):
    dark_region_box = np.where(hr > threshold_value)
    border = ((np.min(dark_region_box[0]), np.max(dark_region_box[0])),
              (np.min(dark_region_box[1]), np.max(dark_region_box[1])),
              (np.min(dark_region_box[2]), np.max(dark_region_box[2])))

    return border


def remove_border(label, hr, interp, border):
    hr = hr[border[0][0]:border[0][1], border[1][0]:border[1][1], border[2][0]:border[2][1]]
    label = label[border[0][0]:border[0][1], border[1][0]:border[1][1], border[2][0]:border[2][1]]
    interp = interp[border[0][0]:border[0][1], border[1][0]:border[1][1], border[2][0]:border[2][1]]
    return label, hr, interp
        
        
def norm_and_interp(hr, LR, order, up_scale):
    # Normalization by the max valeur of LR image
    max_value = np.max(LR)
    normalized_reference_image = hr/max_value
    normalized_low_resolution_image = LR/max_value
        
    # Cubic Interpolation     
    interpolated_image = scipy.ndimage.zoom(normalized_low_resolution_image, zoom=up_scale, order=order)
    
    return interpolated_image, normalized_reference_image


def add_noise(lr, per_cent_val_max):
    
    sigma = per_cent_val_max*np.max(lr)

    lr = lr + np.random.normal(scale=sigma, size=lr.shape)
    
    lr[lr < 0] = 0

    return lr


def create_lr_hr_label(reference_name, label_name, new_resolution):

    # Read the reference SR image
    if reference_name.endswith('.nii.gz'):
        reference_instance = NIFTIReader(reference_name)
    elif os.path.isdir(reference_name):
        reference_instance = DICOMReader(reference_name)

    reference_image = reference_instance.get_np_array()

    # Read the labels image
    if label_name.endswith('.nii.gz'):
        label_instance = NIFTIReader(label_name)
    elif os.path.isdir(label_name):
        label_instance = DICOMReader(label_name)

    label_image = label_instance.get_np_array()

    constant = 2*np.sqrt(2*np.log(2))
    # As Greenspan et al. (Full_width_at_half_maximum : slice thickness)
    sigma_blur = new_resolution/constant

    # Get resolution to scaling factor
    up_scale = tuple(itemb/itema for itema, itemb in zip(reference_instance.itk_image.GetSpacing(), new_resolution))

    # Modcrop to scale factor
    reference_image = modcrop3D(reference_image, up_scale)

    label_image = modcrop3D(label_image, up_scale)


    # ===== Generate input LR image =====
    # Blurring
    BlurReferenceImage = scipy.ndimage.filters.gaussian_filter(reference_image, sigma=sigma_blur)

    print('Generating LR images with the resolution of ', new_resolution)

    # Down sampling
    low_resolution_image = scipy.ndimage.zoom(BlurReferenceImage, zoom=(1/float(idxScale) for idxScale in up_scale),
                                              order=0)

    return low_resolution_image, reference_image, label_image, up_scale


def change_contrast(hr, lr, power):
    hr = hr**power
    lr = lr**power
    
    return hr, lr
