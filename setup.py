import setuptools
from os.path import abspath

with open(abspath("README.md"), encoding='utf-8') as fh:
    long_description = fh.read()

with open(abspath("requirements.txt")) as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="SegSRGAN",
    version="2.1.8",
    author="Clément Cazorla", #Quentin Delannoy, Guillaume Dollé, Nicolas Passat, François Rousseau",
    author_email="clement.cazorla@univ-reims.fr", # quentin.delannoy@univ-reims.fr, guillaume.dolle@univ-reims.fr, nicolas.passat@univ-reims.fr, francois.rousseau@imt-atlantique.fr",
    description="Segmentation and super resolution GAN network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/koopa31/SegSRGAN",
    packages=setuptools.find_packages(),
    install_requires=requirements,
)
