import setuptools
from os.path import abspath


with open(abspath("SegSRGAN/README.md"), encoding='utf-8') as fh:
    long_description = fh.read()

with open(abspath("SegSRGAN/requirements.txt")) as f:
    requirements = f.read().splitlines()


setuptools.setup(
    name="SegSRGAN",
    version="2.0",
    author="Clément Cazorla",
    author_email="clement.cazorla@univ-reims.fr",
    description="Segmentation and super resolution GAN network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/koopa31/SegSRGAN",
    packages=setuptools.find_packages(),
    install_requires=requirements,
)
