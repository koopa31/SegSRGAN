.PHONY: testsuite

all: help

# Parameters to set : 

python="3.7"
package_test = "false"

# Parameters automatically set :

Test_folder_path=$(CURDIR)
	

# Comand : 
create_venv_and_install_SegSRGAN : 
	bash $(Test_folder_path)/bash_scipt_file/create_venv_and_install_segsrgan.sh $(python) $(Test_folder_path) $(package_test)
test_job_model_help: 
	bash $(Test_folder_path)/bash_scipt_file/test_job_model_help.sh $(python) $(Test_folder_path) $(package_test)
remove_venv: 
	rm -rf $(Test_folder_path)/venv_$(python) $(package_test)
remove_result : 
	bash $(Test_folder_path)/bash_scipt_file/Remove_result_folder.sh $(Test_folder_path) $(package_test)
test_seg: 	
	bash $(Test_folder_path)/bash_scipt_file/Test_seg.sh $(python) $(Test_folder_path) $(package_test)
test_job_model_on_image_absolute:
	bash $(Test_folder_path)/bash_scipt_file/test_job_model_on_image.sh $(python) $(Test_folder_path) true $(package_test)
test_job_model_on_image_relatif:
	bash $(Test_folder_path)/bash_scipt_file/test_job_model_on_image.sh $(python) $(Test_folder_path) false $(package_test)
test_training:
	bash $(Test_folder_path)/bash_scipt_file/test_training.sh $(python) $(Test_folder_path) $(package_test)


	
	
	



	

	
	






