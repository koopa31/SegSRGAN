import os

import progressbar
import sys

import numpy as np
import SimpleITK as sitk
import scipy.ndimage
from ast import literal_eval as make_tuple

s = os.path.abspath(__file__).split("/")
wd = "/".join(s[0:(len(s) - 1)])
os.chdir(wd)
sys.path.insert(0, os.getcwd() + '/utils')

from utils3d import shave3D
from utils3d import pad3D
from SegSRGAN import SegSRGAN
from ImageReader import NIFTIReader
from ImageReader import DICOMReader


class SegSRGAN_test(object):

    def __init__(self, weights, patch1, patch2, patch3, is_conditional, resolution=0):

        self.patch1 = patch1
        self.patch2 = patch2
        self.patch3 = patch3
        self.prediction = None
        self.SegSRGAN = SegSRGAN(ImageRow=patch1,
                                 ImageColumn=patch2,
                                 ImageDepth=patch3, is_conditional=is_conditional)
        self.GeneratorModel = self.SegSRGAN.generator_model_for_pred()
        self.GeneratorModel.load_weights(weights, by_name=True)
        self.generator = self.SegSRGAN.generator()
        self.is_conditional = is_conditional
        self.resolution = resolution
        self.ResTensor = np.expand_dims(np.expand_dims(np.ones([patch1, patch2, patch3]) * self.resolution, axis=0),
                                        axis=0)

    def get_patch(self):

        return self.patch

    def test_by_patch(self, TestImage, step=1, by_batch=False):

        # Init temp
        Height, Width, Depth = np.shape(TestImage)

        TempHRImage = np.zeros_like(TestImage)
        TempSeg = np.zeros_like(TestImage)
        WeightedImage = np.zeros_like(TestImage)

        # if is_conditional is set to True we predict on the image AND the resolution
        if self.is_conditional is True:
            if not by_batch:

                i = 0
                bar = progressbar.ProgressBar(maxval=len(np.arange(0, Height - self.patch1 + 1, step)) * len(
                    np.arange(0, Width - self.patch2 + 1, step)) * len(np.arange(0, Depth - self.patch3 + 1, step))).start()
                print('Patch=', self.patch1)
                print('Step=', step)
                for idx in range(0, Height - self.patch1 + 1, step):
                    for idy in range(0, Width - self.patch2 + 1, step):
                        for idz in range(0, Depth - self.patch3 + 1, step):
                            # Cropping image
                            TestPatch = TestImage[idx:idx + self.patch1, idy:idy + self.patch2, idz:idz + self.patch3]
                            ImageTensor = TestPatch.reshape(1, 1, self.patch1, self.patch2, self.patch3).astype(np.float32)
                            PredictPatch = self.generator.predict([ImageTensor, self.ResTensor], batch_size=1)

                            # Adding
                            TempHRImage[idx:idx + self.patch1, idy:idy + self.patch2,
                            idz:idz + self.patch3] += PredictPatch[0, 0, :, :, :]
                            TempSeg[idx:idx + self.patch1, idy:idy + self.patch2, idz:idz + self.patch3] += PredictPatch[0,
                                                                                                            1, :, :, :]
                            WeightedImage[idx:idx + self.patch1, idy:idy + self.patch2,
                            idz:idz + self.patch3] += np.ones_like(PredictPatch[0, 0, :, :, :])

                            i += 1

                            bar.update(i)
            else:

                Height = TestImage.shape[0]
                Width = TestImage.shape[1]
                Depth = TestImage.shape[2]

                patch1 = self.patch1
                patch2 = self.patch2
                patch3 = self.patch3

                patches = np.array([[TestImage[idx:idx + patch1, idy:idy + patch2, idz:idz + patch3]] for idx in
                                    range(0, Height - patch1 + 1, step) for idy in range(0, Width - patch2 + 1, step) for
                                    idz in range(0, Depth - patch3 + 1, step)])

                indice_patch = np.array([(idx, idy, idz) for idx in range(0, Height - patch1 + 1, step) for idy in
                                         range(0, Width - patch2 + 1, step) for idz in range(0, Depth - patch3 + 1, step)])

                pred = self.generator.predict(patches, batch_size=patches.shape[0])

                Weight = np.zeros_like(TestImage)
                TempHRImage = np.zeros(TestImage)
                TempSeg = np.zeros(TestImage)

                for i in range(indice_patch.shape[0]):
                    TempHRImage[indice_patch[i][0]:indice_patch[i][0] + patch1,
                    indice_patch[i][1]:indice_patch[i][1] + patch2, indice_patch[i][2]:indice_patch[i][2] + patch3] += pred[
                                                                                                                       i, 0,
                                                                                                                       :, :,
                                                                                                                       :]
                    TempSeg[indice_patch[i][0]:indice_patch[i][0] + patch1, indice_patch[i][1]:indice_patch[i][1] + patch2,
                    indice_patch[i][2]:indice_patch[i][2] + patch3] += pred[i, 1, :, :, :]
                    Weight[indice_patch[i][0]:indice_patch[i][0] + patch1, indice_patch[i][1]:indice_patch[i][1] + patch2,
                    indice_patch[i][2]:indice_patch[i][2] + patch3] + np.ones_like(
                        Weight[indice_patch[i][0]:indice_patch[i][0] + patch1,
                        indice_patch[i][1]:indice_patch[i][1] + patch2, indice_patch[i][2]:indice_patch[i][2] + patch3])
        else:
            if not by_batch:

                i = 0
                bar = progressbar.ProgressBar(maxval=len(np.arange(0, Height - self.patch1 + 1, step)) * len(
                    np.arange(0, Width - self.patch2 + 1, step)) * len(
                    np.arange(0, Depth - self.patch3 + 1, step))).start()
                print('Patch=', self.patch1)
                print('Step=', step)
                for idx in range(0, Height - self.patch1 + 1, step):
                    for idy in range(0, Width - self.patch2 + 1, step):
                        for idz in range(0, Depth - self.patch3 + 1, step):
                            # Cropping image
                            TestPatch = TestImage[idx:idx + self.patch1, idy:idy + self.patch2, idz:idz + self.patch3]
                            ImageTensor = TestPatch.reshape(1, 1, self.patch1, self.patch2, self.patch3).astype(
                                np.float32)
                            PredictPatch = self.generator.predict(ImageTensor, batch_size=1)

                            # Adding
                            TempHRImage[idx:idx + self.patch1, idy:idy + self.patch2,
                            idz:idz + self.patch3] += PredictPatch[0, 0, :, :, :]
                            TempSeg[idx:idx + self.patch1, idy:idy + self.patch2,
                            idz:idz + self.patch3] += PredictPatch[0,
                                                      1, :, :, :]
                            WeightedImage[idx:idx + self.patch1, idy:idy + self.patch2,
                            idz:idz + self.patch3] += np.ones_like(PredictPatch[0, 0, :, :, :])

                            i += 1

                            bar.update(i)
            else:

                Height = TestImage.shape[0]
                Width = TestImage.shape[1]
                Depth = TestImage.shape[2]

                patch1 = self.patch1
                patch2 = self.patch2
                patch3 = self.patch3

                patches = np.array([[TestImage[idx:idx + patch1, idy:idy + patch2, idz:idz + patch3]] for idx in
                                    range(0, Height - patch1 + 1, step) for idy in range(0, Width - patch2 + 1, step)
                                    for
                                    idz in range(0, Depth - patch3 + 1, step)])

                indice_patch = np.array([(idx, idy, idz) for idx in range(0, Height - patch1 + 1, step) for idy in
                                         range(0, Width - patch2 + 1, step) for idz in
                                         range(0, Depth - patch3 + 1, step)])

                pred = self.generator.predict(patches, batch_size=patches.shape[0])

                Weight = np.zeros_like(TestImage)
                TempHRImage = np.zeros(TestImage)
                TempSeg = np.zeros(TestImage)

                for i in range(indice_patch.shape[0]):
                    TempHRImage[indice_patch[i][0]:indice_patch[i][0] + patch1,
                    indice_patch[i][1]:indice_patch[i][1] + patch2,
                    indice_patch[i][2]:indice_patch[i][2] + patch3] += pred[
                                                                       i, 0,
                                                                       :, :,
                                                                       :]
                    TempSeg[indice_patch[i][0]:indice_patch[i][0] + patch1,
                    indice_patch[i][1]:indice_patch[i][1] + patch2,
                    indice_patch[i][2]:indice_patch[i][2] + patch3] += pred[i, 1, :, :, :]
                    Weight[indice_patch[i][0]:indice_patch[i][0] + patch1,
                    indice_patch[i][1]:indice_patch[i][1] + patch2,
                    indice_patch[i][2]:indice_patch[i][2] + patch3] + np.ones_like(
                        Weight[indice_patch[i][0]:indice_patch[i][0] + patch1,
                        indice_patch[i][1]:indice_patch[i][1] + patch2, indice_patch[i][2]:indice_patch[i][2] + patch3])
        # Weight sum of patches
        print('Done !')
        EstimatedHR = TempHRImage / WeightedImage
        EstimatedSegmentation = TempSeg / WeightedImage

        return (EstimatedHR, EstimatedSegmentation)


def segmentation(input_file_path, step, NewResolution, path_output_cortex, path_output_HR, weights_path, patch=None,
                 spline_order=3, by_batch=False, is_conditional=False):
    # TestFile = path de l'image en entree
    # high_resolution = tuple des resolutions (par axe)

    # Check resolution
    if np.isscalar(NewResolution):
        NewResolution = (NewResolution, NewResolution, NewResolution)
    else:
        if len(NewResolution) != 3:
            raise AssertionError('Not support this resolution !')

    # Read low-resolution image
    if input_file_path.endswith('.nii.gz'):
        image_instance = NIFTIReader(input_file_path)
    elif os.path.isdir(input_file_path):
        image_instance = DICOMReader(input_file_path)

    TestImage = image_instance.get_np_array()
    TestImageMinValue = float(np.min(TestImage))
    TestImageMaxValue = float(np.max(TestImage))
    TestImageNorm = TestImage / TestImageMaxValue


    resolution = image_instance.get_resolution()
    ITK_image = image_instance.ITK_image

    # Check scale factor type
    UpScale = tuple(itema / itemb for itema, itemb in zip(ITK_image.GetSpacing(), NewResolution))

    # spline interpolation
    InterpolatedImage = scipy.ndimage.zoom(TestImageNorm,
                                           zoom=UpScale,
                                           order=spline_order)

    if patch is not None:

        print("patch given")

        patch1 = patch2 = patch3 = int(patch)

        border = (
        int((InterpolatedImage.shape[0] - int(patch)) % step), int((InterpolatedImage.shape[1] - int(patch)) % step),
        int((InterpolatedImage.shape[2] - int(patch)) % step))

        border_to_add = (step - border[0], step - border[1], step - border[2])

        # padd border
        PaddedInterpolatedImage = pad3D(InterpolatedImage, border_to_add)  # remove border of the image

    else:
        border = (
        int(InterpolatedImage.shape[0] % 4), int(InterpolatedImage.shape[1] % 4), int(InterpolatedImage.shape[2] % 4))
        border_to_add = (step - border[0], step - border[1], step - border[2])

        PaddedInterpolatedImage = pad3D(InterpolatedImage, border_to_add)  # remove border of the image

        Height, Width, Depth = np.shape(PaddedInterpolatedImage)
        patch1 = Height
        patch2 = Width
        patch3 = Depth

    # Loading weights
    SegSRGAN_test_instance = SegSRGAN_test(weights_path, patch1, patch2, patch3, is_conditional, resolution)

    # GAN
    print("Testing : ")
    EstimatedHRImage, EstimatedCortex = SegSRGAN_test_instance.test_by_patch(PaddedInterpolatedImage, step=step,
                                                                             by_batch=by_batch)
    # parcours de l'image avec le patch

    # Padding
    # on fait l'operation de padding a l'envers
    PaddedEstimatedHRImage = shave3D(EstimatedHRImage, border_to_add)
    EstimatedCortex = shave3D(EstimatedCortex, border_to_add)

    # SR image
    EstimatedHRImageInverseNorm = PaddedEstimatedHRImage * TestImageMaxValue
    EstimatedHRImageInverseNorm[
        EstimatedHRImageInverseNorm <= TestImageMinValue] = TestImageMinValue  # Clear negative value
    OutputImage = sitk.GetImageFromArray(np.swapaxes(EstimatedHRImageInverseNorm, 0, 2))
    OutputImage.SetSpacing(NewResolution)
    OutputImage.SetOrigin(ITK_image.GetOrigin())
    OutputImage.SetDirection(ITK_image.GetDirection())

    sitk.WriteImage(OutputImage, path_output_HR)

    # Cortex segmentation
    OutputCortex = sitk.GetImageFromArray(np.swapaxes(EstimatedCortex, 0, 2))
    OutputCortex.SetSpacing(NewResolution)
    OutputCortex.SetOrigin(ITK_image.GetOrigin())
    OutputCortex.SetDirection(ITK_image.GetDirection())

    sitk.WriteImage(OutputCortex, path_output_cortex)

    return "Segmentation Done"
