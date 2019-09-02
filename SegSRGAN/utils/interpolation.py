import numpy as np
import scipy
import SimpleITK as sitk

class Interpolation():
    def __init__(self, normalized_low_resolution_image, upscale, order, interp):
        self.upscale = upscale
        self.order = order
        self.interp = interp
        self.normalized_low_resolution_image = normalized_low_resolution_image

    def get_interpolated_image(self):
        if self.interp == 'scipy':
            interpolated_image = self.get_scipy_interpolation()
        elif self.interp == 'sitk':
            interpolated_image = self.get_sitk_interpolation()
        return interpolated_image

    def get_scipy_interpolation(self):
        normalized_low_resolution_image = self.normalized_low_resolution_image
        interpolated_image = scipy.ndimage.zoom(normalized_low_resolution_image, zoom=self.upscale, order=self.order)
        return interpolated_image

    def get_sitk_interpolation(self):
        normalized_low_resolution_image = sitk.GetImageFromArray(self.normalized_low_resolution_image)
        self.upscale = [int(i) for i in self.upscale]
        self.upscale = swapList(self.upscale)
        interp = sitk.Expand(normalized_low_resolution_image, list(self.upscale), sitk.sitkBSpline)
        interp = sitk.GetArrayFromImage(interp)
        return interp


def swapList(newList):
    size = len(newList)

    # Swapping
    temp = newList[0]
    newList[0] = newList[size - 1]
    newList[size - 1] = temp

    return newList