import SimpleITK as sitk
import numpy as np
from abc import ABC, abstractmethod


class ImageReader(ABC):
    """Class to read MRI images"""
    def __init__(self, image_path):
        self.image_path = image_path
        self.itk_image = self.set_itk_image()

    @abstractmethod
    def get_np_array(self):
        pass

    @abstractmethod
    def get_resolution(self):
        pass

    @abstractmethod
    def set_itk_image(self):
        pass


class NIFTIReader(ImageReader):
    """Class to read NIFTI MRI images"""
    def get_np_array(self):
        """Return the NIFTI image as a Numpy array"""

        test_image = np.swapaxes(sitk.GetArrayFromImage(self.itk_image), 0, 2).astype('float32')
        return test_image

    def get_resolution(self):
        """Return the spacing between the slices"""

        reader = sitk.ImageFileReader()

        reader.SetFileName(self.image_path)
        reader.LoadPrivateTagsOn();

        reader.ReadImageInformation();

        resolution = float(reader.GetMetaData('pixdim[3]'))

        return resolution

    def set_itk_image(self):
        """Set ITK image object"""
        itk_image = sitk.ReadImage(self.image_path)
        return itk_image


class DICOMReader(ImageReader):
    """Class to read DICOM MRI images"""
    def get_np_array(self):
        """Return the DICOM image as a Numpy array"""

        reader = sitk.ImageSeriesReader()

        dicom_names = reader.GetGDCMSeriesFileNames(self.image_path)
        reader.SetFileNames(dicom_names)

        image = reader.Execute()
        test_image = np.swapaxes(sitk.GetArrayFromImage(image), 0, 2).astype('float32')
        return test_image

    def get_resolution(self):
        """Return the spacing between the slices"""

        reader = sitk.ImageSeriesReader()

        dicom_names = reader.GetGDCMSeriesFileNames(self.image_path)
        filename = dicom_names[0]
        reader = sitk.ImageFileReader()

        reader.SetFileName(filename)
        reader.LoadPrivateTagsOn();

        reader.ReadImageInformation();

        resolution = reader.GetMetaData('0018|0088')

        return float(resolution)

    def set_itk_image(self):
        """Set ITK image object"""
        reader = sitk.ImageSeriesReader()

        dicom_names = reader.GetGDCMSeriesFileNames(self.image_path)
        filename = dicom_names[0]
        itk_image = sitk.ReadImage(filename)
        return itk_image
