import numpy as np


class Normalization():
    
    def __init__(self, LR, hr=None,type_of_normalization="max"):
        # hr=None correspond to the normalization done in test (only LR is available)
        self.hr = hr
        self.LR = LR
        self.type_of_normalization=type_of_normalization

    # Pairs of function corresponding to the normalization by the max value.
    def normalization_by_max(self):
    
        max_value = np.max(self.LR)
        normalized_low_resolution_image = self.LR / max_value
        
        if self.hr is None :

            return normalized_low_resolution_image,None

        normalized_reference_image = self.hr / max_value
        
        return normalized_low_resolution_image, normalized_reference_image
	        
    def denormalization_result_by_max(self,normalized_SR):
        # Used to put the scale of the SR result in the same range as the initial image.
        max_value = np.max(self.LR)
        SR=normalized_SR*max_value
        return SR
    
    # applying function in main :
    
    def get_normalized_image(self):

        if (self.type_of_normalization=="max") :
    
            normalized_low_resolution_image, normalized_reference_image = self.normalization_by_max()
    
        return normalized_low_resolution_image,normalized_reference_image

    def get_denormalized_result_image(self,normalized_SR):
    
        if self.type_of_normalization=="max" :
    
            SR = self.denormalization_result_by_max(normalized_SR)
    
        return SR

