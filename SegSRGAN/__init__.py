import os
from .utils.download import download_weights
from pathlib import Path
parent=Path(__file__).resolve().parent

name = "SegSRGAN"
weights_path = os.path.join(str(parent),"weights")

download_weights( weights_path )
