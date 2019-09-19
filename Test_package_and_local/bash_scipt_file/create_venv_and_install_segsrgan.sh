#!/bin/bash


virtualenv -p python$1 $2/venv_$1

source $2/venv_$1/bin/activate

echo "Pip is :"$(which pip)

echo "Python is :"$(which python)

if [[ $3 == "true" ]]
then
	pip install SegSRGAN
else 
	SCRIPT=$(readlink -f $0)
	parent="$(dirname $SCRIPT)"
	parent_2="$(dirname $parent)"
	parent_3="$(dirname $parent_2)"
	requierement_file=$parent_3"/requirements.txt"
	pip install -r $requierement_file
fi
