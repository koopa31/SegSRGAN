import Function_for_application_test_python3
import pandas as pd
import glob
import os



###### Wrnings : 

# Il faut que la script soit appliquer a plus d'un fichier car path_pour_application doit etre une liste

###### Argument :

Descripif_poids="poids intiaux" # string qui apparaitra dans le dossier cree pour stocker les result : code avec + Descripif_poids

path_relatif_poids="/weights/SegSRGAN"

debut_relatif_path="/home/qdelannoy/Segmentation_neural_network/" #Path qui amene au fichier Base_pour_romeo

data=pd.read_csv(debut_relatif_path+"Base_pour_romeo/Meta_data_data_for_romeo.csv")

data=data[::-1] #renversement de l ordre

path_pour_application=debut_relatif_path+data["Path_relatif_pour_romeo"]


ensemble_pas=[]

#ensemble_pas.append([20,20])
#
#ensemble_pas.append([5,20]) #list des step qui vont etre appliquer pour la premiere taille de patch  
#
#ensemble_pas.append([10,40]) #multiplication de la taille du patch par 2, multiplication du pas par 2

ensemble_pas.append([40])

patchs=[128]


####### fonction : 

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


######## Execution : 

ensemble_pas=pd.DataFrame(ensemble_pas,index=patchs)

np_pas_per_patch=ensemble_pas.shape[1]

weights_path =  os.getcwd()+path_relatif_poids




for i in path_pour_application :
    
    for patch in patchs :

        if patch is None : 
            
            i_split=i.split("/")

            path_output = "/".join(i_split[:(len(i_split)-1)])+"/code avec "+Descripif_poids+"/patch "+str(patch)
            
            print(path_output)
               
            if  not os.path.exists(path_output):
                
                createFolder(path_output)
                    
                path_output_cortex = path_output + "/Cortex "+str(patch)+".nii.gz"
                    
                path_output_SR=path_output+"/SR " +str(patch)+".nii.gz"
                
                Function_for_application_test_python3.segmentation(input_file_path=i,
                                                                   step=20,
                                                                   NewResolution=(0.5,0.5,0.5),
                                                                   patch=patch,
                                                                   path_output_cortex=path_output_cortex,
                                                                   path_output_HR=path_output_SR,
                                                                   weights_path=weights_path
                                                                   )
            else :
                
                print("déja calculé")
        else :
    
            for step in ensemble_pas.loc[patch]:

                i_split=i.split("/")
    
                path_output = "/".join(i_split[:(len(i_split)-1)])+"/code avec "+Descripif_poids+"/patch "+str(patch)+" step "+str(step)+" inversion shave padd"
                    
                print(path_output)
                    
                if  not os.path.exists(path_output):
                            
                    createFolder(path_output)
                        
                    path_output_cortex = path_output + "/Cortex patch "+str(patch)+" step "+str(step)+".nii.gz"
                        
                    path_output_SR=path_output+"/SR.nii.gz"+str(patch)+" step "+str(step)+".nii.gz"
                    
                    Function_for_application_test_python3.segmentation(input_file_path=i,
                                                                       step=step,
                                                                       NewResolution=(0.5,0.5,0.5),
                                                                       patch=patch,
                                                                       path_output_cortex=path_output_cortex,
                                                                       path_output_HR=path_output_SR,
                                                                       weights_path=weights_path
                                                                       )
                else :
                        
                    print("déja calculé")
                

