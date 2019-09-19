#!/bin/bash

source $2/venv_$1/bin/activate

echo "Python is :"$(which python)

HR=$(find $2"/Image_for_training/HR/" -name "*.nii.gz")
Label=$(find $2"/Image_for_training/Label/" -name "*.nii.gz")


HR=($HR) #Transform in list
Label=($Label) #Transform in list
Base=(Train Test)

text="HR_image,Label_image,Base"


weights="weights/Perso_without_data_augmentation"

for i in "${!HR[@]}"; do 
  text=$text"\n"${HR[$i]}","${Label[$i]}","${Base[$i]}
done


printf $text >> training.csv



if [[ $3 == "true" ]]
then 
	python_file_path=$2/venv_$1/lib/python$1/site-packages/SegSRGAN/SegSRGAN_training.py

else 
	SCRIPT=$(readlink -f $0)
	parent="$(dirname $SCRIPT)"
	parent_2="$(dirname $parent)"
	parent_3="$(dirname $parent_2)"
	python_file_path=$parent_3"/SegSRGAN/SegSRGAN_training.py"
fi


python $python_file_path -e 1 --new_low_res 0.5 0.5 3 --csv $2/"training.csv" --snapshot_folder $2/"weights_test/" --dice_file $2/"dice.csv" --mse_file $2"/mse.csv" --folder_training_data $2"/tpm" --number_of_disciminator_iteration 1; 


clean_up() {

	rm -rf $2"/tmp/";
	rm $2"training.csv";
	rm -rf $2/"weights_test/";
	rm $2/"dice.csv";
	rm $2/"mse.csv"
}

rm -rf $2"/tmp/";
rm $2"training.csv";
rm -rf $2/"weights_test/";
rm $2/"dice.csv";
rm $2/"mse.csv"

trap clean_up EXIT


