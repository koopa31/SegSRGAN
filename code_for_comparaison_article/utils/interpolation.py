import numpy as np
import scipy
import SimpleITK as sitk

interpolations = {'NearestNeighbor': sitk.sitkNearestNeighbor,
'Linear': sitk.sitkLinear,
'Spline': sitk.sitkBSpline,
'Gaussian': sitk.sitkGaussian,
'Hamming': sitk.sitkHammingWindowedSinc,
'Cosine': sitk.sitkCosineWindowedSinc,
'Welch': sitk.sitkWelchWindowedSinc,
'Lanczos': sitk.sitkLanczosWindowedSinc,
'Blackman':sitk.sitkBlackmanWindowedSinc}


class Interpolation():
    def __init__(self, normalized_low_resolution_image, upscale, order, interp, interpolation_type):
        self.upscale = upscale
        self.order = order
        self.interp = interp
        self.normalized_low_resolution_image = normalized_low_resolution_image
        self.interpolation_type = interpolation_type

    def get_interpolated_image(self, original_LR=None):
        if self.interp == 'scipy':
            interpolated_image, self.upscale = self.get_scipy_interpolation()
        elif self.interp == 'sitk':
            interpolated_image, self.upscale = self.get_sitk_interpolation(original_LR, self.interpolation_type)
        return interpolated_image, self.upscale

    def get_scipy_interpolation(self):
        normalized_low_resolution_image = self.normalized_low_resolution_image
        interpolated_image = scipy.ndimage.zoom(normalized_low_resolution_image, zoom=self.upscale, order=self.order)
        return interpolated_image, self.upscale

    def get_sitk_interpolation(self, original_LR, interpolation_type):
        normalized_low_resolution_image = sitk.GetImageFromArray(np.swapaxes(self.normalized_low_resolution_image, 0, 2))
        if type(original_LR) == type(normalized_low_resolution_image):
            normalized_low_resolution_image.SetSpacing(original_LR.GetSpacing())
            normalized_low_resolution_image.SetOrigin(original_LR.GetOrigin())
            normalized_low_resolution_image.SetDirection(original_LR.GetDirection())
        else:
            normalized_low_resolution_image.SetSpacing(original_LR.itk_image.GetSpacing())
            normalized_low_resolution_image.SetOrigin(original_LR.itk_image.GetOrigin())
            normalized_low_resolution_image.SetDirection(original_LR.itk_image.GetDirection())
        self.upscale = [int(round(i)) for i in self.upscale]
        interp = sitk.Expand(normalized_low_resolution_image, list(self.upscale), interpolations[interpolation_type])
        interp = np.swapaxes(sitk.GetArrayFromImage(interp), 0, 2)
        return interp, self.upscale


def swapList(newList):
    size = len(newList)

    # Swapping
    temp = newList[0]
    newList[0] = newList[size - 1]
    newList[size - 1] = temp

    return newList