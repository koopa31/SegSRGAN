import numpy as np


class Normalization():
    def __init__(self, LR, hr):
        self.hr = hr
        self.LR = LR

    def get_normalized_image(self):
        max_value = np.max(self.LR)
        normalized_reference_image = self.hr / max_value
        normalized_low_resolution_image = self.LR / max_value
        return normalized_low_resolution_image, normalized_reference_image