import setuptools
from os.path import abspath


with open(abspath("SegSRGAN/README.md"), "r") as fh:
    long_description = fh.read()

with open(abspath("SegSRGAN/requirements.txt")) as f:
    requirements = f.read().splitlines()


setuptools.setup(
    name="SegSRGAN",
    version="1.1.2",
    author="Clément Cazorla",
    author_email="clement.cazorla@univ-reims.fr",
    description="Segmentation and super resolution GAN network",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/koopa31/SegSRGAN/tree/develop",
    packages=setuptools.find_packages(),
    install_requires=requirements,
)
