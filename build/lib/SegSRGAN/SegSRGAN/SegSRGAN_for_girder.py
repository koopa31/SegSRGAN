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

s=os.path.abspath(__file__).split("/")

wd="/".join(s[0:(len(s)-1)])


os.chdir(wd)

import numpy as np
import SimpleITK as sitk
import scipy.ndimage
from ast import literal_eval as make_tuple

import sys
sys.path.insert(0, os.getcwd()+'/utils')
from utils3d import shave3D
from utils3d import pad3D
from SegSRGAN import SegSRGAN
import progressbar


class SegSRGAN_test(object):
        def __init__(self,weights,patch1,patch2,patch3):
            
            self.patch1 = patch1
            self.patch2 = patch2
            self.patch3 = patch3
            self.prediction = None
            self.SegSRGAN = SegSRGAN(ImageRow = patch1, 
                                     ImageColumn = patch2, 
                                     ImageDepth = patch3)
            self.GeneratorModel = self.SegSRGAN.generator_model_for_pred()
            self.GeneratorModel.load_weights(weights, by_name=True)
            self.generator = self.SegSRGAN.generator()
            
        def get_patch(self):
            
            return self.patch
            
        def test_by_patch(self,TestImage,step=1):  
            
            # Init temp
            Height,Width,Depth = np.shape(TestImage)
            
            
            TempHRImage = np.zeros_like(TestImage)
            TempSeg = np.zeros_like(TestImage)
            WeightedImage = np.zeros_like(TestImage)
            
            i=0
            bar = progressbar.ProgressBar(maxval=len(np.arange(0,Height-self.patch1+1,step))*len(np.arange(0,Width-self.patch2+1,step))*len(np.arange(0,Depth-self.patch3+1,step))).start()
    
            for idx in range(0,Height-self.patch1+1,step):
                for idy in range(0,Width-self.patch2+1,step):
                    for idz in range(0,Depth-self.patch3+1,step):  
                        
                        # Cropping image
                        TestPatch = TestImage[idx:idx+self.patch1,idy:idy+self.patch2,idz:idz+self.patch3] 
                        ImageTensor = TestPatch.reshape(1,1,self.patch1,self.patch2,self.patch3).astype(np.float32)
                        PredictPatch =  self.generator.predict(ImageTensor, batch_size=1)
                        
                        # Adding
                        TempHRImage[idx:idx+self.patch1,idy:idy+self.patch2,idz:idz+self.patch3] += PredictPatch[0,0,:,:,:]
                        TempSeg [idx:idx+self.patch1,idy:idy+self.patch2,idz:idz+self.patch3] += PredictPatch[0,1,:,:,:]
                        WeightedImage[idx:idx+self.patch1,idy:idy+self.patch2,idz:idz+self.patch3] += np.ones_like(PredictPatch[0,0,:,:,:])
                        
                        i+=1
                        
                        bar.update(i)
                        
                        
            # Weight sum of patches
            print('Done !')
            EstimatedHR = TempHRImage/WeightedImage
            EstimatedSegmentation = TempSeg/WeightedImage
            return (EstimatedHR,EstimatedSegmentation)
    



@girder_job(title='Segmentation haute resolution du cortex cerebral')
@app.task(bind=True)

def segmentation(self,input_file_path,step,spline_order,high_resolution,path_output_cortex,patch=None,):
    
    # TestFile = path de l'image en entree
    # high_resolution = tuple des resolutions (par axe)
        
     # Check resolution
    NewResolution = make_tuple(high_resolution)
    if np.isscalar(NewResolution):
        NewResolution = (NewResolution,NewResolution,NewResolution)
    else:
        if len(NewResolution)!=3:
            raise AssertionError('Not support this resolution !')
            
#    # Check border removing
#    border = make_tuple(args.border)
#    if np.isscalar(border):
#        border = (border,border,border)
#    else:
#        if len(border)!=3:
#            raise AssertionError, 'Not support this border !' 
        
    
    
          
    # Read low-resolution image
    TestNifti = sitk.ReadImage(input_file_path)
    TestImage = np.swapaxes(sitk.GetArrayFromImage(TestNifti),0,2).astype('float32')
    TestImageMinValue = float(np.min(TestImage))
    TestImageMaxValue = float(np.max(TestImage))
    TestImageNorm = TestImage/TestImageMaxValue 
    
    # Check scale factor type
    UpScale = tuple(itema/itemb for itema,itemb in zip(TestNifti.GetSpacing(),NewResolution)) 
    
    
    # spline interpolation 
    InterpolatedImage = scipy.ndimage.zoom(TestImageNorm, 
                                           zoom = UpScale,
                                           order = spline_order) 
    
    if patch is not None : 

        
        patch1 = patch2 = patch3 = int(patch)
        
        border=((InterpolatedImage.shape[0]-int(patch))%step,(InterpolatedImage.shape[1]-int(patch))%step,(InterpolatedImage.shape[2]-int(patch))%step)
    
        # Shave border
        ShavedInterpolatedImage = shave3D(InterpolatedImage, border) # remove border of the image
        
    else :
        border=(InterpolatedImage.shape[0]%4,InterpolatedImage.shape[1]%4,InterpolatedImage.shape[2]%4)
        ShavedInterpolatedImage = shave3D(InterpolatedImage, border) # remove border of the image
        Height,Width,Depth = np.shape(ShavedInterpolatedImage)
        patch1 = Height
        patch2 = Width
        patch3 = Depth
        
    #Weight path : 
    
    weight_path=os.getcwd()+"/weights/SegSRGAN"
    
    # Loading weights
    SegSRGAN_test_instance = SegSRGAN_test(weight_path,patch1,patch2,patch3)
    
    # GAN 
    print("Testing : ")
    EstimatedHRImage, EstimatedCortex  = SegSRGAN_test_instance.test_by_patch(ShavedInterpolatedImage,step=args.step) # parcours de l'image avec le patch
    
    # Padding
    #on fait l'operation de saving a l'envers
    PaddedEstimatedHRImage = pad3D(EstimatedHRImage,border)
    EstimatedCortex = pad3D(EstimatedCortex,border)
    
    # SR image 
    EstimatedHRImageInverseNorm = PaddedEstimatedHRImage*TestImageMaxValue
    EstimatedHRImageInverseNorm[EstimatedHRImageInverseNorm <= TestImageMinValue] = TestImageMinValue    # Clear negative value
    OutputImage = sitk.GetImageFromArray(np.swapaxes(EstimatedHRImageInverseNorm,0,2))
    OutputImage.SetSpacing(NewResolution)
    OutputImage.SetOrigin(TestNifti.GetOrigin())
    OutputImage.SetDirection(TestNifti.GetDirection())
    
    # Cortex segmentation
    OutputCortex = sitk.GetImageFromArray(np.swapaxes(EstimatedCortex,0,2))
    OutputCortex.SetSpacing(NewResolution)
    OutputCortex.SetOrigin(TestNifti.GetOrigin())
    OutputCortex.SetDirection(TestNifti.GetDirection())
    
    return "Segmentation Done"




