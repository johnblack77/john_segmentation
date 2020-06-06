import numpy as np


args = {
    'BATCH_SIZE'            :   5
,   'MAX_EPOCH'             :   300
,   'TrainFineTest'         :   'train'
,   'USE_GPU'               :   True
,   'PRINT_FREQ'            :   100
,   'ROOT_PATH'             :   ''
,   'DATA_PATH'             :   '../data/CamVid/'
,   'TRAIN_LOG_PATH'        :   '../data/trainlog/'
,   'SAVE_MODEL_PATH'       :   '../data/modellog/'
,   'TEST_DATA_PATN'        :   '../data/CamVid/test.txt'
,   'TEST_SAVE_PATH'        :   '../data/testlog/'
,   'SUMMARY_PATH'          :   '../data/summary/'
,   'LOAD_MODEL_PATH'       :   '../data/modellog/0601_10:20:31/58_0601_22:00:01.pkl'
,   'BATCH_TEST_SAVE_PATH'  :   '../data/batchtest/'
,   'ONE_HOTE'              :   False
,   'OPTIMZIER'             :   'Adam'
,   'lr'                    :   1e-4
,   'lr_decay'              :   0.7
,   'weight_decay'          :   0#5e-5
,   'NUM_WORKER'            :   8
,   'PIP_MEMORY'            :   True
,   'NON_BLOCKING'          :   True
,   'CHANNAL_SIZE'          :   3
,   'HIGH_SIZE'             :   304
,   'WIDTH_SIZE'            :   304
,   'NUM_CLASS'             :   12
}


loss_weight = np.array([
      0.2595,
      0.1826,
      4.5640,
      0.1417,
      0.9051,
      0.3826,
      9.6446,
      1.8418,
      0.6823,
      6.2478,
      7.3614,
      1.0974])




Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road_marking = [255, 69, 0]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]




label_colours = np.array([Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

all_categories = ['Sky','Building','Pole', 'Road','Pavement','Tree','SignSymbol','Fence','Car','Pedestrian','Bicyclist', 'Unlabelled']

'''
'0:sky, 1:build, 2:pole, 3:road, 4:Pavement, 5:tree, 6:SignSymbol, 7:Fence, 8:Car, 9:Pedestrian, 10:Bicyclist, 11:unlabel'
'''
