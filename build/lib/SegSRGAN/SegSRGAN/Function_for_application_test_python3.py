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
        self.SegSRGAN = SegSRGAN(image_row=patch1,
                                 image_column=patch2,
                                 image_depth=patch3, is_conditional=is_conditional)
        self.generator_model = self.SegSRGAN.generator_model_for_pred()
        self.generator_model.load_weights(weights, by_name=True)
        self.generator = self.SegSRGAN.generator()
        self.is_conditional = is_conditional
        self.resolution = resolution
        self.res_tensor = np.expand_dims(np.expand_dims(np.ones([patch1, patch2, patch3]) * self.resolution, axis=0),
                                        axis=0)

    def get_patch(self):
        """

        :return:
        """
        return self.patch

    def test_by_patch(self, test_image, step=1, by_batch=False):
        """

        :param test_image: Image to be tested
        :param step: step
        :param by_batch: to enable by batch processing
        :return:
        """
        # Init temp
        height, width, depth = np.shape(test_image)

        temp_hr_image = np.zeros_like(test_image)
        temp_seg = np.zeros_like(test_image)
        weighted_image = np.zeros_like(test_image)

        # if is_conditional is set to True we predict on the image AND the resolution
        if self.is_conditional is True:
            if not by_batch:

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
                            predict_patch = self.generator.predict([image_tensor, self.res_tensor], batch_size=1)

                            # Adding
                            temp_hr_image[idx:idx + self.patch1, idy:idy + self.patch2,
                            idz:idz + self.patch3] += predict_patch[0, 0, :, :, :]
                            temp_seg[idx:idx + self.patch1, idy:idy + self.patch2, idz:idz + self.patch3] += \
                                predict_patch[0, 1, :, :, :]
                            weighted_image[idx:idx + self.patch1, idy:idy + self.patch2,
                            idz:idz + self.patch3] += np.ones_like(predict_patch[0, 0, :, :, :])

                            i += 1

                            bar.update(i)
            else:

                height = test_image.shape[0]
                width = test_image.shape[1]
                depth = test_image.shape[2]

                patch1 = self.patch1
                patch2 = self.patch2
                patch3 = self.patch3

                patches = np.array([[test_image[idx:idx + patch1, idy:idy + patch2, idz:idz + patch3]] for idx in
                                    range(0, height - patch1 + 1, step) for idy in range(0, width - patch2 + 1, step)
                                    for idz in range(0, depth - patch3 + 1, step)])

                indice_patch = np.array([(idx, idy, idz) for idx in range(0, height - patch1 + 1, step) for idy in
                                         range(0, width - patch2 + 1, step) for idz in range(0, depth - patch3 + 1,
                                                                                             step)])

                pred = self.generator.predict(patches, batch_size=patches.shape[0])

                weight = np.zeros_like(test_image)
                temp_hr_image = np.zeros(test_image)
                temp_seg = np.zeros(test_image)

                for i in range(indice_patch.shape[0]):
                    temp_hr_image[indice_patch[i][0]:indice_patch[i][0] + patch1,
                    indice_patch[i][1]:indice_patch[i][1] + patch2, indice_patch[i][2]:indice_patch[i][2] + patch3] += pred[
                                                                                                                       i, 0,
                                                                                                                       :, :,
                                                                                                                       :]
                    temp_seg[indice_patch[i][0]:indice_patch[i][0] + patch1, indice_patch[i][1]:indice_patch[i][1] + patch2,
                    indice_patch[i][2]:indice_patch[i][2] + patch3] += pred[i, 1, :, :, :]
                    weight[indice_patch[i][0]:indice_patch[i][0] + patch1, indice_patch[i][1]:indice_patch[i][1] + patch2,
                    indice_patch[i][2]:indice_patch[i][2] + patch3] + np.ones_like(
                        weight[indice_patch[i][0]:indice_patch[i][0] + patch1,
                        indice_patch[i][1]:indice_patch[i][1] + patch2, indice_patch[i][2]:indice_patch[i][2] + patch3])
        else:
            if not by_batch:

                i = 0
                bar = progressbar.ProgressBar(maxval=len(np.arange(0, height - self.patch1 + 1, step)) * len(
                    np.arange(0, width - self.patch2 + 1, step)) * len(
                    np.arange(0, depth - self.patch3 + 1, step))).start()
                print('Patch=', self.patch1)
                print('Step=', step)
                for idx in range(0, height - self.patch1 + 1, step):
                    for idy in range(0, width - self.patch2 + 1, step):
                        for idz in range(0, depth - self.patch3 + 1, step):
                            # Cropping image
                            test_patch = test_image[idx:idx + self.patch1, idy:idy + self.patch2, idz:idz + self.patch3]
                            image_tensor = test_patch.reshape(1, 1, self.patch1, self.patch2, self.patch3).astype(
                                np.float32)
                            predict_patch = self.generator.predict(image_tensor, batch_size=1)

                            # Adding
                            temp_hr_image[idx:idx + self.patch1, idy:idy + self.patch2,
                            idz:idz + self.patch3] += predict_patch[0, 0, :, :, :]
                            temp_seg[idx:idx + self.patch1, idy:idy + self.patch2,
                            idz:idz + self.patch3] += predict_patch[0,
                                                      1, :, :, :]
                            weighted_image[idx:idx + self.patch1, idy:idy + self.patch2,
                            idz:idz + self.patch3] += np.ones_like(predict_patch[0, 0, :, :, :])

                            i += 1

                            bar.update(i)
            else:

                height = test_image.shape[0]
                width = test_image.shape[1]
                depth = test_image.shape[2]

                patch1 = self.patch1
                patch2 = self.patch2
                patch3 = self.patch3

                patches = np.array([[test_image[idx:idx + patch1, idy:idy + patch2, idz:idz + patch3]] for idx in
                                    range(0, height - patch1 + 1, step) for idy in range(0, width - patch2 + 1, step)
                                    for
                                    idz in range(0, depth - patch3 + 1, step)])

                indice_patch = np.array([(idx, idy, idz) for idx in range(0, height - patch1 + 1, step) for idy in
                                         range(0, width - patch2 + 1, step) for idz in
                                         range(0, depth - patch3 + 1, step)])

                pred = self.generator.predict(patches, batch_size=patches.shape[0])

                weight = np.zeros_like(test_image)
                temp_hr_image = np.zeros(test_image)
                temp_seg = np.zeros(test_image)

                for i in range(indice_patch.shape[0]):
                    temp_hr_image[indice_patch[i][0]:indice_patch[i][0] + patch1,
                    indice_patch[i][1]:indice_patch[i][1] + patch2,
                    indice_patch[i][2]:indice_patch[i][2] + patch3] += pred[
                                                                       i, 0,
                                                                       :, :,
                                                                       :]
                    temp_seg[indice_patch[i][0]:indice_patch[i][0] + patch1,
                    indice_patch[i][1]:indice_patch[i][1] + patch2,
                    indice_patch[i][2]:indice_patch[i][2] + patch3] += pred[i, 1, :, :, :]
                    weight[indice_patch[i][0]:indice_patch[i][0] + patch1,
                    indice_patch[i][1]:indice_patch[i][1] + patch2,
                    indice_patch[i][2]:indice_patch[i][2] + patch3] + np.ones_like(
                        weight[indice_patch[i][0]:indice_patch[i][0] + patch1,
                        indice_patch[i][1]:indice_patch[i][1] + patch2, indice_patch[i][2]:indice_patch[i][2] + patch3])
        # weight sum of patches
        print('Done !')
        estimated_hr = temp_hr_image / weighted_image
        estimated_segmentation = temp_seg / weighted_image

        return estimated_hr, estimated_segmentation


def segmentation(input_file_path, step, new_resolution, path_output_cortex, path_output_hr, weights_path, patch=None,
                 spline_order=3, by_batch=False, is_conditional=False):
    """

    :param input_file_path: path of the image to be super resolved and segmented
    :param step: the shifting step for the patches
    :param new_resolution: the new z-resolution we want for the output image
    :param path_output_cortex: output path of the segmented cortex
    :param path_output_hr: output path of the super resolution output image
    :param weights_path: the path of the file which contains the pre-trained weights for the neural network
    :param patch: the size of the patches
    :param spline_order: for the interpolation
    :param by_batch: to enable the by-batch processing
    :param is_conditional: to perform a conditional GAN on the LR image resolution
    :return:
    """
    # TestFile = path de l'image en entree
    # high_resolution = tuple des resolutions (par axe)

    # Check resolution
    if np.isscalar(new_resolution):
        new_resolution = (new_resolution, new_resolution, new_resolution)
    else:
        if len(new_resolution) != 3:
            raise AssertionError('Not support this resolution !')

    # Read low-resolution image
    if input_file_path.endswith('.nii.gz'):
        image_instance = NIFTIReader(input_file_path)
    elif os.path.isdir(input_file_path):
        image_instance = DICOMReader(input_file_path)

    test_image = image_instance.get_np_array()
    test_imageMinValue = float(np.min(test_image))
    test_imageMaxValue = float(np.max(test_image))
    test_imageNorm = test_image / test_imageMaxValue


    resolution = image_instance.get_resolution()
    itk_image = image_instance.itk_image

    # Check scale factor type
    up_scale = tuple(itema / itemb for itema, itemb in zip(itk_image.GetSpacing(), new_resolution))

    # spline interpolation
    interpolated_image = scipy.ndimage.zoom(test_imageNorm,
                                           zoom=up_scale,
                                           order=spline_order)

    if patch is not None:

        print("patch given")

        patch1 = patch2 = patch3 = int(patch)

        border = (
        int((interpolated_image.shape[0] - int(patch)) % step), int((interpolated_image.shape[1] - int(patch)) % step),
        int((interpolated_image.shape[2] - int(patch)) % step))

        border_to_add = (step - border[0], step - border[1], step - border[2])

        # padd border
        padded_interpolated_image = pad3D(interpolated_image, border_to_add)  # remove border of the image

    else:
        border = (
        int(interpolated_image.shape[0] % 4), int(interpolated_image.shape[1] % 4), int(interpolated_image.shape[2] %
                                                                                        4))
        border_to_add = (4 - border[0], 4 - border[1], 4 - border[2])

        padded_interpolated_image = pad3D(interpolated_image, border_to_add)  # remove border of the image

        height, width, depth = np.shape(padded_interpolated_image)
        patch1 = height
        patch2 = width
        patch3 = depth

    # Loading weights
    segsrgan_test_instance = SegSRGAN_test(weights_path, patch1, patch2, patch3, is_conditional, resolution)

    # GAN
    print("Testing : ")
    estimated_hr_image, estimated_cortex = segsrgan_test_instance.test_by_patch(padded_interpolated_image, step=step,
                                                                             by_batch=by_batch)
    # parcours de l'image avec le patch

    # Padding
    # on fait l'operation de padding a l'envers
    padded_estimated_hr_image = shave3D(estimated_hr_image, border_to_add)
    estimated_cortex = shave3D(estimated_cortex, border_to_add)

    # SR image
    estimated_hr_imageInverseNorm = padded_estimated_hr_image * test_imageMaxValue
    estimated_hr_imageInverseNorm[
        estimated_hr_imageInverseNorm <= test_imageMinValue] = test_imageMinValue  # Clear negative value
    output_image = sitk.GetImageFromArray(np.swapaxes(estimated_hr_imageInverseNorm, 0, 2))
    output_image.SetSpacing(new_resolution)
    output_image.SetOrigin(itk_image.GetOrigin())
    output_image.SetDirection(itk_image.GetDirection())

    sitk.WriteImage(output_image, path_output_hr)

    # Cortex segmentation
    output_cortex = sitk.GetImageFromArray(np.swapaxes(estimated_cortex, 0, 2))
    output_cortex.SetSpacing(new_resolution)
    output_cortex.SetOrigin(itk_image.GetOrigin())
    output_cortex.SetDirection(itk_image.GetDirection())

    sitk.WriteImage(output_cortex, path_output_cortex)

    return "Segmentation Done"
