#!/bin/bash

source $2/venv_$1/bin/activate

echo "Python is :"$(which python)

yourfilenames=$(find $2"/Image_for_testing/" -name "*.nii.gz" -type f )

echo "first_file:"$yourfilenames | awk '{print $2}'

for eachfile in $yourfilenames
do
   	if [[ $eachfile == *"Result"* ]]
	then
  		:
	else 
		img_path=$eachfile
	fi
done


printf $img_path >> $2/job_model.csv

echo $img_path

weights="weights/Perso_with_constrast_0.5_and_noise_0.03_val_max"


if [[ $4 == "true" ]]
then 
	python_file_path=$2/venv_$1/lib/python$1/site-packages/SegSRGAN/job_model.py
	weight_path="weights_without_augmentation"
	if $3

	then 
		weight_path=$2/venv_$1/lib/python$1/site-packages/SegSRGAN/$weights
        
	fi
else 
	SCRIPT=$(readlink -f $0)
	parent="$(dirname $SCRIPT)"
	parent_2="$(dirname $parent)"
	parent_3="$(dirname $parent_2)"
	python_file_path=$parent_3"/SegSRGAN/job_model.py"
	weight_path=$parent_3"/data/"$weights
fi


echo "Used weights : "$weight_path
echo "File : "$python_file_path

python $python_file_path --path $2/job_model.csv\
			--patch "128,200"\
			--step "100 128,150 200"\
			--result_folder_name "weights_with_augmentation"\
			--weights_path $weight_path


rm $2/job_model.csv
