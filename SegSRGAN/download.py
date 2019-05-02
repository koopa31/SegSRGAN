import os

import wget
import stat
import requests


def download_weights():
    # We get the content of the weights folder
    z = requests.get('https://api.github.com/repos/koopa31/SegSRGAN/contents/SegSRGAN/SegSRGAN/weights?ref=develop')
    contents = z.json()
    weights_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SegSRGAN/weights')
    # Creation of weights folder if it does not already exist
    if os.path.isdir(weights_path) is False:
        os.mkdir(weights_path, mode=0o777)
    # Downloading of the files
    for content in contents:
        if os.path.isfile(os.path.join(weights_path, content['name'])) is False:
            print("Processing %s" % content['path'])
            wget.download(content['download_url'], out=weights_path)
