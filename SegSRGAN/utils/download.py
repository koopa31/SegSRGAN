import os
import wget
import stat
import requests
import warnings

def check_internet():
    url='http://www.google.com/'
    timeout=5
    try:
        _ = requests.get(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        return False

# Download precalculated weight files used for the machine learning algorithm
# and available from the repository.
def download_weights( weights_path ):
	if check_internet():
		# We get the content of the weights folder.
		z = requests.get('https://api.github.com/repos/koopa31/SegSRGAN/contents/data/weights?ref=master')
		contents = z.json()
		# Creation of weights folder if it does not already exist.
		if os.path.isdir(weights_path) is False:
			print(weights_path)
			os.mkdir(weights_path, mode=0o777)
		# Downloading of the files.
		for content in contents:
			if os.path.isfile(os.path.join(weights_path, content['name'])) is False:
				print("Processing %s" % content['path'])
				wget.download(content['download_url'], out=weights_path)
	else :
		warnings.warn("The internet connection doesn't work so the local weights set cannot be updated with the remote one. May cause issue of non existing file")
	