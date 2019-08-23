.PHONY: testsuite

all: help

pkg: clean
	@python3 -m pip install --user --upgrade setuptools wheel
	@python3 setup.py sdist bdist_wheel

pkg-upload: pkg
	@python3 -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

test:
	@python3 testsuite/seg.py

clean:
	@rm -rf build
	@rm -rf dist
	@rm -rf SegSRGAN.egg-info

dry-clean:
	@git clean -xdi

help:
	@printf "\
Usage: make <target>\n\
\n\
target:\n\
    pkg 	Create the pip package\n\
    pkg-ulpoad 	Push the pip package\n\
    test	Run the testsuite\n\
    clean	Remove temporary files\n\
    dry-clean	Remove all untracked files\n\
\n"

