from pathlib import Path
import numpy as np
import natsort
from Collect_Paths import Collect_Paths
import ezc3d
from ezc3d import c3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import FileLink
from matplotlib.animation import FuncAnimation
import os, sys
sys.path.append('/home/giuliamartinelli/Documents/Repositories/DECA/AMCParser')
sys.path.append('/home/giuliamartinelli/Documents/Code/smplx')
import amc_parser as amc
from human_body_prior.body_model.body_model import BodyModel
import torch
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import pandas as pd
import numpy as np

train_dataset = pd.read_csv('/home/giuliamartinelli/Documents/Repositories/DECA/dataset/train_data.csv')
val_dataset = pd.read_csv('/home/giuliamartinelli/Documents/Repositories/DECA/dataset/validation_data.csv')
test_dataset = pd.read_csv('/home/giuliamartinelli/Documents/Repositories/DECA/dataset/test_data.csv')
train_files = train_dataset['File c3d'].values
validation_files = val_dataset['File c3d'].values
test_files = test_dataset['File c3d'].values


bm_fname = "/home/giuliamartinelli/Documents/Code/smplify-x/models/smplx/SMPLX_NEUTRAL.npz"
bm = BodyModel(bm_fname=bm_fname, num_betas=10)
faces = c2c(bm.f)

train_data, validation_data, test_data = [], [], []

train_skeleton_motions, validation_skeleton_motions, test_skeleton_motions = [], [], []

train_smpl_motions, validation_smpl_motions, test_smpl_motions = [], [], []

for i in range(len(train_dataset['File c3d'].values)):
    c = c3d(train_files[i])
    train_joints = amc.parse_asf((train_dataset['File asf'].values).tolist()[i])
    train_motions = amc.parse_amc((train_dataset['File amc'].values).tolist()[i])
    bdata = np.load((train_dataset['File smpl'].values).tolist()[i])
    if (train_dataset['Len amc'].values).tolist()[i] > (train_dataset['Len c3d'].values).tolist()[i]:
        len_motion = (train_dataset['Len c3d'].values).tolist()[i]
    else:
        len_motion = (train_dataset['Len amc'].values).tolist()[i]
    body_parms = {
        'root_orient': torch.Tensor(bdata['root_orient']), # controls the global root orientation
        'pose_body': torch.Tensor(bdata['pose_body']), # controls the body
        'pose_hand': torch.Tensor(bdata['pose_hand']), # controls the finger articulation
        'trans': torch.Tensor(bdata['trans']), # controls the global body position
        'betas': torch.Tensor(np.repeat(bdata['betas'][:10][np.newaxis], repeats=(train_dataset['Len c3d'].values).tolist()[i], axis=0)), # controls the body shape. Body shape is static
    }
    body_pose_hand = bm(**{k:v for k,v in body_parms.items() if k in ['root_orient','pose_body', 'betas', 'pose_hand','trans']})
    
    for frame in range(len_motion):
        tranpose_data = c['data']['points'][:3,:41,frame].transpose()
        train_data.append(tranpose_data/1000)
        train_joints['root'].set_motion(train_motions[frame])
        c_joints = train_joints['root'].to_dict()
        skeletons_coordinates = []
        for index, joint in enumerate(c_joints.values()):
            skeletons_coordinates.append(joint.coordinate[:,0]*0.056444)
        train_skeleton_motions.append(np.array(skeletons_coordinates))
        smpl_joints = c2c(body_pose_hand.Jtr[frame])
        train_smpl_motions.append(smpl_joints)
        

for i in range(len(validation_files)):
    c = c3d(validation_files[i])
    validation_joints = amc.parse_asf((val_dataset['File asf'].values).tolist()[i])
    validation_motions = amc.parse_amc((val_dataset['File amc'].values).tolist()[i])
    bdata = np.load((val_dataset['File smpl'].values).tolist()[i])

    if (val_dataset['Len amc'].values).tolist()[i] > (val_dataset['Len c3d'].values).tolist()[i]:
        len_motion = (val_dataset['Len c3d'].values).tolist()[i]
    else:
        len_motion = (val_dataset['Len amc'].values).tolist()[i]

    body_parms = {
        'root_orient': torch.Tensor(bdata['root_orient']), # controls the global root orientation
        'pose_body': torch.Tensor(bdata['pose_body']), # controls the body
        'pose_hand': torch.Tensor(bdata['pose_hand']), # controls the finger articulation
        'trans': torch.Tensor(bdata['trans']), # controls the global body position
        'betas': torch.Tensor(np.repeat(bdata['betas'][:10][np.newaxis], repeats=(val_dataset['Len c3d'].values).tolist()[i], axis=0)), # controls the body shape. Body shape is static
    }
    body_pose_hand = bm(**{k:v for k,v in body_parms.items() if k in ['root_orient','pose_body', 'betas', 'pose_hand','trans']})
    
    for frame in range(len_motion):
        tranpose_data = c['data']['points'][:3,:41,frame].transpose()
        validation_data.append(tranpose_data/1000)
        validation_joints['root'].set_motion(validation_motions[frame])
        c_joints = validation_joints['root'].to_dict()
        skeletons_coordinates = []
        for index, joint in enumerate(c_joints.values()):
            skeletons_coordinates.append(joint.coordinate[:,0]*0.056444)
        validation_skeleton_motions.append(np.array(skeletons_coordinates))
        smpl_joints = c2c(body_pose_hand.Jtr[frame])
        validation_smpl_motions.append(smpl_joints)

for i in range(len(test_files)):
    c = c3d(test_files[i])
    test_joints = amc.parse_asf((test_dataset['File asf'].values).tolist()[i])
    test_motions = amc.parse_amc((test_dataset['File amc'].values).tolist()[i])
    bdata = np.load((test_dataset['File smpl'].values).tolist()[i])
    if (test_dataset['Len amc'].values).tolist()[i] > (test_dataset['Len c3d'].values).tolist()[i]:
        len_motion = (test_dataset['Len c3d'].values).tolist()[i]
    else:
        len_motion = (test_dataset['Len amc'].values).tolist()[i]

    body_parms = {
        'root_orient': torch.Tensor(bdata['root_orient']), # controls the global root orientation
        'pose_body': torch.Tensor(bdata['pose_body']), # controls the body
        'pose_hand': torch.Tensor(bdata['pose_hand']), # controls the finger articulation
        'trans': torch.Tensor(bdata['trans']), # controls the global body position
        'betas': torch.Tensor(np.repeat(bdata['betas'][:10][np.newaxis], repeats=(test_dataset['Len c3d'].values).tolist()[i], axis=0)), # controls the body shape. Body shape is static
    }
    body_pose_hand = bm(**{k:v for k,v in body_parms.items() if k in ['root_orient','pose_body', 'betas', 'pose_hand','trans']})
    
    for frame in range(len_motion):
        tranpose_data = c['data']['points'][:3,:41,frame].transpose()
        test_data.append(tranpose_data/1000)
        test_joints['root'].set_motion(test_motions[frame])
        c_joints = test_joints['root'].to_dict()
        skeletons_coordinates = []
        for index, joint in enumerate(c_joints.values()):
            skeletons_coordinates.append(joint.coordinate[:,0]*0.056444)
        test_skeleton_motions.append(np.array(skeletons_coordinates))
        smpl_joints = c2c(body_pose_hand.Jtr[frame])
        test_smpl_motions.append(smpl_joints)

train_array = np.array(train_data)
val_array = np.array(validation_data)
test_array = np.array(test_data)

skel_train = np.array(train_skeleton_motions)
skel_val = np.array(validation_skeleton_motions)
skel_test = np.array(test_skeleton_motions)

smpl_train = np.array(train_smpl_motions)
smpl_val = np.array(validation_smpl_motions)
smpl_test = np.array(test_smpl_motions)


np.savez("/home/giuliamartinelli/Documents/Repositories/DECA/dataset/train.npz", c3d=train_array, skeleton=skel_train, smpl=smpl_train)
np.savez("/home/giuliamartinelli/Documents/Repositories/DECA/dataset/validation.npz", c3d=val_array, skeleton=skel_val, smpl=smpl_val)
np.savez("/home/giuliamartinelli/Documents/Repositories/DECA/dataset/test.npz", c3d=test_array, skeleton=skel_test, smpl=smpl_test)

print('fine')