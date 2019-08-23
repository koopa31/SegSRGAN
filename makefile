.PHONY: testsuite

all: help

pkg: clean
	@python3 -m pip install --user --upgrade setuptools wheel
	@python3 setup.py sdist bdist_wheel
test:
	@python3 testsuite/seg.py

clean:
	@rm -rf ./dist
	@rm -rf ./build
	@rm -rf ./SegSRGAN.egg-info

help:
	@printf "\
Usage: make <target>\n\
\n\
target:\n\
    pkg 	Create the pip package\n\
    test	Run the testsuite\n\
    clean	Remove temporary files\n\
\n"

