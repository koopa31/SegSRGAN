import numpy as np
import scipy
import SimpleITK as sitk

class Interpolation():
    def __init__(self, normalized_low_resolution_image, upscale, order, interp):
        self.upscale = upscale
        self.order = order
        self.interp = interp
        self.normalized_low_resolution_image = normalized_low_resolution_image

    def get_interpolated_image(self, original_LR=None):
        if self.interp == 'scipy':
            interpolated_image, self.upscale = self.get_scipy_interpolation()
        elif self.interp == 'sitk':
            interpolated_image, self.upscale = self.get_sitk_interpolation(original_LR)
        return interpolated_image, self.upscale

    def get_scipy_interpolation(self):
        normalized_low_resolution_image = self.normalized_low_resolution_image
        interpolated_image = scipy.ndimage.zoom(normalized_low_resolution_image, zoom=self.upscale, order=self.order)
        return interpolated_image, self.upscale

    def get_sitk_interpolation(self, original_LR):
        normalized_low_resolution_image = sitk.GetImageFromArray(np.swapaxes(self.normalized_low_resolution_image, 0, 2))
        normalized_low_resolution_image.SetSpacing(original_LR.itk_image.GetSpacing())
        normalized_low_resolution_image.SetOrigin(original_LR.itk_image.GetOrigin())
        normalized_low_resolution_image.SetDirection(original_LR.itk_image.GetDirection())
        self.upscale = [round(i) for i in self.upscale]
        interp = sitk.Expand(normalized_low_resolution_image, list(self.upscale), sitk.sitkBSpline)
        interp = np.swapaxes(sitk.GetArrayFromImage(interp), 0, 2)
        return interp, self.upscale


def swapList(newList):
    size = len(newList)

    # Swapping
    temp = newList[0]
    newList[0] = newList[size - 1]
    newList[size - 1] = temp

    return newList