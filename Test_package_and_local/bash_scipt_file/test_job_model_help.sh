#!/bin/bash

source $2/venv_$1/bin/activate

echo "Python is :"$(which python)



if [[ $3 == "true" ]]
then 
	python_file_path=$2/venv_$1/lib/python$1/site-packages/SegSRGAN/job_model.py

else 
	SCRIPT=$(readlink -f $0)
	parent="$(dirname $SCRIPT)"
	parent_2="$(dirname $parent)"
	parent_3="$(dirname $parent_2)"
	python_file_path=$parent_3"/SegSRGAN/job_model.py"
fi



python $python_file_path --help
