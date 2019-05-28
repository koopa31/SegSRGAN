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


def shave3D(image, border):
    '''
    Remove border of an image
    Note: Array of Python in interval [i:j] will affect from i to j-1
    -----
    image : 3d array
    border : tuple, indicates border is removed
    Example : border = (10,10,10)
    
    '''
    if np.isscalar(border):
        print('attention border scalaire')
        image = image[border:image.shape[0]-border, border:image.shape[1]-border, border:image.shape[2]-border]
    else:
        axis_0 = []
        if border[0] % 2 == 0:
            axis_0 = [border[0]/2, border[0]/2]
        else:
            axis_0 = [(border[0]-1)/2, (border[0]+1)/2]
        
        axis_1 = []
        if border[1] % 2 == 0:
            axis_1 = [border[1]/2, border[1]/2]
        else:
            axis_1 = [(border[1]-1)/2, (border[1]+1)/2]
        
        axis_2 = []
        if border[2] % 2 == 0:
            axis_2 = [border[2]/2, border[2]/2]
        else:
            axis_2 = [(border[2]-1)/2, (border[2]+1)/2]
            
        axis_0 = np.array(axis_0,dtype=int)
        axis_1 = np.array(axis_1,dtype=int)
        axis_2 = np.array(axis_2,dtype=int)
        
        image = image[axis_0[0]:image.shape[0]-axis_0[1], axis_1[0]:image.shape[1]-axis_1[1],
                      axis_2[0]:image.shape[2]-axis_2[1]]
        
    return image


def pad3D(image, border):
    '''
    Remove border of an image
    Note: Array of Python in interval [i:j] will affect from i to j-1
    -----
    image : 3d array
    border : tuple, indicates border is removed
    Example : border = (10,10,10)
    
    '''
    if np.isscalar(border):
        image = image[border:image.shape[0]-border,border:image.shape[1]-border, border:image.shape[2]-border]
        print('attention border scalaire')
    else:
        axis_0 = []
        if border[0] % 2 == 0:
            axis_0 = [border[0]/2, border[0]/2]
        else :
            axis_0 = [(border[0]-1)/2, (border[0]+1)/2]
        
        axis_1 = []
        if border[1] % 2 == 0:
            axis_1 = [border[1]/2, border[1]/2]
        else :
            axis_1 = [(border[1]-1)/2, (border[1]+1)/2]
        
        axis_2 = []
        if border[2] % 2 == 0:
            axis_2 = [border[2]/2, border[2]/2]
        else:
            axis_2 = [(border[2]-1)/2, (border[2]+1)/2]
        
        axis_0 = np.array(axis_0, dtype=int)
        axis_1 = np.array(axis_1, dtype=int)
        axis_2 = np.array(axis_2, dtype=int)
        
        image = np.pad(image, pad_width=((axis_0[0], axis_0[1]),(axis_1[0],axis_1[1]),(axis_2[0],axis_2[1])), 
                       mode='constant')
        
    return image


def imadjust3D(image, new_range=None):
    """
        More detail about formula : https://en.wikipedia.org/wiki/Normalization_(image_processing)
        ----
        image : 3d array
        new_range : new range of value
        Example : new_range = [0,1]
    """
    Min = np.min(image)
    Max = np.max(image)
    newMin = new_range[0]
    newMax = new_range[1]
    temp = (newMax - newMin) / float(Max - Min)
    image = ((image - Min) * temp + newMin)
    return image 


def modcrop3D(img, modulo):
    import math
    img = img[0:int(img.shape[0] - math.fmod(img.shape[0], modulo[0])), 
              0:int(img.shape[1] - math.fmod(img.shape[1], modulo[1])), 
              0:int(img.shape[2] - math.fmod(img.shape[2], modulo[2]))]
    return img
