#!/bin/bash

source $2/venv_$1/bin/activate

echo "Python is :"$(which python)


source $2/venv_$1/bin/activate

echo "Python is :"$(which python)

yourfilenames=$(find $2"/Image_for_testing/" -name "*.nii.gz" -type f )


for eachfile in $yourfilenames
do
   	if [[ $eachfile == *"Result"* ]]
	then
  		:
	else 
		img_path=$eachfile
	fi
done

parent_folder="$(dirname $img_path)"

result_folder=$parent_folder"/Result_segmentation_function/"

mkdir $result_folder

python $2/Python_function/seg.py -i $img_path
