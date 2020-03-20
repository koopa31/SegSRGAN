import glob 
import sys
import numpy as np
from keras.optimizers import Adam
from tensorflow.python.client import device_lib
from keras import losses
from keras.utils import multi_gpu_model
import time
import pandas as pd
import os
import argparse

sys.path.insert(0,os.path.split(os.path.split(__file__)[0])[0])



from utils.u_net import UNet
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# Attention à faire un dossier pour train et un pour test.

def train(path_save_npy,snapshot_folder,training_epoch,path_csv):
    
    path_save_npy_training_data = path_save_npy+"/Train/"
    path_save_npy_testing_data = path_save_npy+"/Test/"
    
    g_train = glob.glob(path_save_npy_training_data+"*")
    g_test = glob.glob(path_save_npy_testing_data+"*")
    
    nb_mini_batch_train = len(g_train)/2
    nb_mini_batch_test = len(g_test)/2
    
    os.makedirs(snapshot_folder)
    
    
    net = UNet((1,64,64,64), out_ch=1, start_ch=28, depth=2, inc_rate=2., activation='relu', 
    		 dropout=0, batchnorm=True, maxpool=False, upconv=True, residual=False)
    
    num_gpu = len(get_available_gpus())
    
    print("number of gpus : " + str(num_gpu))
    
    
    if  num_gpu > 1 :
    
        net_multi_gpu = multi_gpu_model(net, gpus=num_gpu, cpu_merge=False)
        print("Model duplicated on : " + str(num_gpu) + " GPUS")
    else:
        net_multi_gpu = net
        print("Model apply on CPU or single GPU")
    
    
    net_multi_gpu.compile(Adam(),loss=[losses.binary_crossentropy])
    
    df_dice = pd.DataFrame(index=np.arange(1, training_epoch + 1), columns=["Dice"])
    
    for EpochIndex in range(1, training_epoch + 1):
        
        for i in range(int(nb_mini_batch_train)) : 
            
            training_input = np.load(path_save_npy_training_data+"Datas_mini_batch_"+str(i)+".npy")
            training_output = np.load(path_save_npy_training_data+"Label_mini_batch_"+str(i)+".npy")
            
        
            loss = net_multi_gpu.train_on_batch([training_input],[training_output])
            
            print("Iter " + str(i) + " [A loss : " + str(loss) + "]")
        
            
        for i in range(int(nb_mini_batch_test)) : 
            
                    VP = []
                    Pos_pred = []
                    Pos_label = []
        
                    t1 = time.time()
        
        
                    TestLabels = np.load(path_save_npy_testing_data+"Label_mini_batch_"+str(i)+".npy")
                    TestDatas = np.load(path_save_npy_testing_data+"Datas_mini_batch_"+str(i)+".npy")
                    
                    pred = net.predict(TestDatas)
                    
                    VP.append(np.sum((pred > 0.5) & (TestLabels== 1)))
        
                    Pos_pred.append(np.sum(pred > 0.5))
        
                    Pos_label.append(np.sum(TestLabels))
        
        Dice = (2 * np.sum(VP)) / (np.sum(Pos_pred) + np.sum(Pos_label))
        
        print("Epoche " + str(EpochIndex) + " [Test Dice : " + str(Dice) + "]")
        
        net.save_weights(snapshot_folder + 'u-net_epoche' + str(EpochIndex))
        
        df_dice.loc[EpochIndex, "Dice"] = Dice       
        
        df_dice.to_csv(path_csv)
        
parser = argparse.ArgumentParser()
parser.add_argument("-path_save_npy", "--path_save_npy", type=str, help="Path of npy files")
parser.add_argument("-snapshot_folder", "--snapshot_folder", type=str, help="Path for saving weights file")
parser.add_argument("-training_epoch", "--training_epoch", type=int, help="Number of training epoch")
parser.add_argument("-csv_file", "--csv_file", type=str, help="path and name of csv file ex : /home/dice.csv")    

args = parser.parse_args()
               
train(args.path_save_npy,args.snapshot_folder,args.training_epoch,args.csv_file)            