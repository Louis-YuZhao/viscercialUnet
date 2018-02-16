# -*- coding: utf-8 -*-
"""
Post Processing after Unet.
"""
#%%
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
#%%
preThreshold = 0.5
groundThreshold = 0.02
leave_one_out_file = 'TrainingDataWbCT'

organ = '187_gallbladder'
#CTwb = 18
#organ = '170_pancreas'
#CTwb = 19
#organ = '29193_first_lumbar_vertebra'
#CTwb = 19
#organ = '30324_right_adrenal_gland'
#CTwb = 14
#organ = '30325_left_adrenal_gland'
#CTwb = 15

#%%
def WriteListtoFile(filelist, filename):
    with open(filename, 'w') as f:
        for i in filelist:
            f.write(i+'\n')
    return 1

def diceComputing():        
    import CompareThePreandtruth as CTP
    data_path = '/media/data/louis/ProgramWorkResult/VisercialUnet/'
    ThreeDImageDir = os.path.join (data_path, 'Pred3D')
    groundTruthDir = os.path.join (data_path, leave_one_out_file, organ +'_Linear_Labelpatch')
    predictInput = ThreeDImageDir + '/FileList.txt'
    groundTruthInput = groundTruthDir + '/FileList.txt'
    predictOutput = os.path.join(data_path, 'Pred3DMod')
    if not os.path.exists(predictOutput):
        subprocess.call('mkdir ' + '-p ' + predictOutput, shell=True)
#    groundTruthOutput = '/media/data/louis/ProgramWorkResult/VisercialUnet/GT3DMod/'
    
    dicorestat = CTP.CompareThePreandTruth(predictInput, groundTruthInput)
    dicorestat.readPredictImagetoList()
    dicorestat.readgroundTruthtoList()
    dicorestat.predictModification(predictOutput, preThreshold)
#    dicorestat.groundTruthModification(groundTruthOutput, groundThreshold)
    diceScore = dicorestat.diceScoreStatistics()

    wbCTDiceScore = diceScore[:CTwb]
    dice_Statistics = {}
    dice_Statistics['mean'] = np.mean(wbCTDiceScore)
    dice_Statistics['std'] = np.std(wbCTDiceScore)
    dice_Statistics['max'] = np.amax(wbCTDiceScore)
    dice_Statistics['min'] = np.amin(wbCTDiceScore)
    print dice_Statistics    
    
    CTceDiceScore = diceScore[CTwb:]
    dice_Statistics = {}
    dice_Statistics['mean'] = np.mean(CTceDiceScore)
    dice_Statistics['std'] = np.std(CTceDiceScore)
    dice_Statistics['max'] = np.amax(CTceDiceScore)
    dice_Statistics['min'] = np.amin(CTceDiceScore)
    print dice_Statistics 

def diceComputing_full():        
    import CompareThePreandtruth as CTP
    data_path = '/media/data/louis/ProgramWorkResult/VisercialUnet/'
    ThreeDImageDir = os.path.join (data_path, 'Pred3D')
    groundTruthDir = os.path.join (data_path, leave_one_out_file, organ +'_Linear_Labelpatch')
    predictInput = ThreeDImageDir + '/FileList.txt'
    groundTruthInput = groundTruthDir + '/FileList.txt'
    predictOutput = os.path.join(data_path, 'Pred3DMod')
    if not os.path.exists(predictOutput):
        subprocess.call('mkdir ' + '-p ' + predictOutput, shell=True)
#    groundTruthOutput = '/media/data/louis/ProgramWorkResult/VisercialUnet/GT3DMod/'
    
    dicorestat = CTP.CompareThePreandTruth(predictInput, groundTruthInput)
    dicorestat.readPredictImagetoList()
    dicorestat.readgroundTruthtoList()
    dicorestat.predictModification(predictOutput, preThreshold)
#    dicorestat.groundTruthModification(groundTruthOutput, groundThreshold)
    dicorestat.diceScoreStatistics()



def showlosscurve():
    tempStore = './tempData' 
#    loss = np.load(os.path.join(tempStore,'loss.npy'))
#    val_loss = np.load(os.path.join(tempStore,'val_loss.npy'))
    loss = np.load(os.path.join(tempStore,'10000011_1_CT_loss.npy'))
    val_loss = np.load(os.path.join(tempStore,'10000011_1_CT_val_loss.npy')) 
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['loss', 'val_loss'])
    plt.show()

if __name__ == '__main__':
#    diceComputing()
    diceComputing_full()
    showlosscurve()