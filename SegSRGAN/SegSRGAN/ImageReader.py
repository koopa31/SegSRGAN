import SimpleITK as sitk
import numpy as np
from abc import ABC, abstractmethod


class ImageReader(ABC):
    def __init__(self, image_path):
        self.image_path = image_path
        self.ITK_image = self.set_ITK_image()

    @abstractmethod
    def get_np_array(self):
        pass

    @abstractmethod
    def get_resolution(self):
        pass

    @abstractmethod
    def set_ITK_image(self):
        pass


class NIFTIReader(ImageReader):

    def get_np_array(self):
        """Return the NIFTI image as a Numpy array"""

        test_image = np.swapaxes(sitk.GetArrayFromImage(self.ITK_image), 0, 2).astype('float32')
        return test_image

    def get_resolution(self):
        """Return the spacing between the slices"""

        reader = sitk.ImageFileReader()

        reader.SetFileName(self.image_path)
        reader.LoadPrivateTagsOn();

        reader.ReadImageInformation();

        resolution = float(reader.GetMetaData('pixdim[3]'))

        return resolution

    def set_ITK_image(self):
        """Set ITK image object"""
        ITK_image = sitk.ReadImage(self.image_path)
        return ITK_image


class DICOMReader(ImageReader):

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

        return resolution

    def set_ITK_image(self):
        """Set ITK image object"""
        reader = sitk.ImageSeriesReader()

        dicom_names = reader.GetGDCMSeriesFileNames(self.image_path)
        filename = dicom_names[0]
        ITK_image = sitk.ReadImage(filename)
        return ITK_image
