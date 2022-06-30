from attr import validate
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import os, sys
import itertools
import torch
import natsort
import glob
sys.path.append('/home/giuliamartinelli/Documents/Repositories/DECA/AMCParser')
import amc_parser as amc


class Collect_Paths():
    def __init__(self, BASE_DIR_C3D = None, BASE_DIR_SKEL = None,BASE_DIR_SMPL = None):
        self.BASE_DIR_C3D = BASE_DIR_C3D
        self.BASE_DIR_SKEL = BASE_DIR_SKEL
        self.BASE_DIR_SMPL = BASE_DIR_SMPL
        self.files_c3d = list(natsort.natsorted(self.BASE_DIR_C3D.glob('*/subjects/*/*.c3d')))
        self.files_amc = list(natsort.natsorted(self.BASE_DIR_SKEL.glob('subjects/*/*.amc')))
        self.files_npz = list(natsort.natsorted(self.BASE_DIR_SMPL.glob('*/*.npz')))
        self.filename_npz = []
        self.filename_c3d = []
        self.filename_amc = []
        for npz in list(self.BASE_DIR_SMPL.glob('*/neutral_stagei.npz')):
            self.files_npz.remove(npz)
        for file in self.files_npz:
            file_str = str(file)
            self.filename_npz.append(file_str.split('/')[-1].split('_stageii.npz')[0])
        print("len smpl files",len(self.filename_npz))
        for file in self.files_amc:
            file_str = str(file)
            self.filename_amc.append(file_str.split('/')[-1].split('.amc')[0])
        print("len amc files",len(self.filename_amc))
        for file in self.files_c3d:
            file_str = str(file)
            self.filename_c3d.append(file_str.split('/')[-1].split('.c3d')[0])
        print("len c3d files",len(self.filename_c3d))

        plus_c3d = natsort.natsorted(list(set(self.filename_c3d)-set(self.filename_amc)))
        plus_amc = natsort.natsorted(list(set(self.filename_amc)-set(self.filename_c3d)))


        for plus1 in plus_c3d:
            self.filename_c3d.remove(plus1)
            self.files_c3d.remove(list(self.BASE_DIR_C3D.glob('*/subjects/*/'+plus1 + '.c3d'))[0])

        for plus2 in plus_amc:
            self.filename_amc.remove(plus2)
            self.files_amc.remove(list(self.BASE_DIR_SKEL.glob('subjects/*/' + plus2 + '.amc'))[0])

        plus_c3d = natsort.natsorted(list(set(self.filename_c3d)-set(self.filename_npz)))
        plus_npz = natsort.natsorted(list(set(self.filename_npz)-set(self.filename_c3d)))


        for plus1 in plus_c3d:
            self.filename_c3d.remove(plus1)
            self.files_c3d.remove(list(BASE_DIR_C3D.glob('*/subjects/*/'+ plus1 + '.c3d'))[0])
            self.filename_amc.remove(plus1)
            self.files_amc.remove(list(BASE_DIR_SKEL.glob('subjects/*/' + plus1 + '.amc'))[0])

        for plus2 in plus_npz:
            self.filename_npz.remove(plus2)
            self.files_npz.remove(list(BASE_DIR_SMPL.glob('*/'+ plus2 + '_stageii.npz'))[0])

        print("len c3d after files",len(self.files_c3d))
        print("len amc after files",len(self.files_amc))
        print("len smpl after files",len(self.files_npz))

        self.subject = []
        for file in self.filename_c3d:
            if file.split('_')[0] not in self.subject:
                self.subject.append(file.split('_')[0])
        
        self.subject = natsort.natsorted(self.subject)
        print("len subject",len(self.subject))


        self.train_skel = []
        self.train_npz = []
        self.validate_skel = []
        self.validate_npz = []
        self.test_skel = []
        self.test_npz = []

        datasets_motion = pd.DataFrame({'path': self.files_c3d})
        datasets_motion['Subject'] = datasets_motion['path'].map(lambda x: x.parent.stem)
        datasets_motion['Activity'] = datasets_motion['path'].map(lambda x: x.stem.split('_')[-1].lower())

        datasets_amc = pd.DataFrame({'path': self.files_amc})
        datasets_amc['Subject'] = datasets_amc['path'].map(lambda x: x.parent.stem)
        datasets_amc['Activity'] = datasets_amc['path'].map(lambda x: x.stem.split('_')[-1].lower())
        datasets_amc['asf_path'] = datasets_amc['path'].map(lambda x: x.parent / (x.parent.stem + '.asf'))

        datasets_npz = pd.DataFrame({'path': self.files_npz})
        datasets_npz['Subject'] = datasets_npz['path'].map(lambda x: x.parent.stem)

        train_len = int(len(self.subject)*0.6)
        train_index, test_index = torch.utils.data.random_split(self.subject, [train_len, len(self.subject) - train_len])
        train_subj = natsort.natsorted(list(self.subject[i] for i in train_index.indices))
        test_set = natsort.natsorted(list(self.subject[i] for i in test_index.indices))
        val_len = int(len(test_set)*0.5)
        val_index, test_index = torch.utils.data.random_split(test_set, [val_len, len(test_set) - val_len])
        val_subj = natsort.natsorted(list(test_set[i] for i in val_index.indices))
        test_subj = natsort.natsorted(list(test_set[i] for i in test_index.indices))

        self.train_c3d = []
        self.validate_c3d = []
        self.test_c3d = []
        self.train_amc = []
        self.validate_amc = []
        self.test_amc = []
        self.train_npz = []
        self.validate_npz = []
        self.test_npz = []
        
        for train, validate, test in zip(train_subj,val_subj,test_subj):
            for index in np.where(datasets_motion['Subject']==train)[0]:
                self.train_c3d.append(datasets_motion['path'].values[index])
            for index in np.where(datasets_motion['Subject']==validate)[0]:
                self.validate_c3d.append(datasets_motion['path'].values[index])
            for index in np.where(datasets_motion['Subject']==test)[0]:   
                self.test_c3d.append(datasets_motion['path'].values[index])

            for index in np.where(datasets_amc['Subject']==train)[0]:
                self.train_amc.append(datasets_amc['path'].values[index])
            for index in np.where(datasets_amc['Subject']==validate)[0]:
                self.validate_amc.append(datasets_amc['path'].values[index])
            for index in np.where(datasets_amc['Subject']==test)[0]:
                self.test_amc.append(datasets_amc['path'].values[index])

            for index in np.where(datasets_npz['Subject']==train)[0]:
                self.train_npz.append(datasets_npz['path'].values[index])
            for index in np.where(datasets_npz['Subject']==validate)[0]:
                self.validate_npz.append(datasets_npz['path'].values[index])
            for index in np.where(datasets_npz['Subject']==test)[0]:
                self.test_npz.append(datasets_npz['path'].values[index])

    def get_c3d_data(self):
        train_motion = pd.DataFrame({'path': self.train_c3d})
        train_motion['Subject'] = train_motion['path'].map(lambda x: x.parent.stem)
        train_motion['Activity'] = train_motion['path'].map(lambda x: x.stem.split('_')[-1].lower())
        validate_motion = pd.DataFrame({'path': self.validate_c3d})
        validate_motion['Subject'] = validate_motion['path'].map(lambda x: x.parent.stem)
        validate_motion['Activity'] = validate_motion['path'].map(lambda x: x.stem.split('_')[-1].lower())
        test_motion = pd.DataFrame({'path': self.test_c3d})
        test_motion['Subject'] = test_motion['path'].map(lambda x: x.parent.stem)
        test_motion['Activity'] = test_motion['path'].map(lambda x: x.stem.split('_')[-1].lower())
        return train_motion, validate_motion, test_motion

    def get_amc_data(self):
        train_skel = pd.DataFrame({'path': self.train_amc})
        train_skel['Subject'] = train_skel['path'].map(lambda x: x.parent.stem)
        train_skel['Activity'] = train_skel['path'].map(lambda x: x.stem.split('_')[-1].lower())
        train_skel['asf_path'] = train_skel['path'].map(lambda x: x.parent / (x.parent.stem + '.asf'))
        validate_skel = pd.DataFrame({'path': self.validate_amc})
        validate_skel['Subject'] = validate_skel['path'].map(lambda x: x.parent.stem)
        validate_skel['Activity'] = validate_skel['path'].map(lambda x: x.stem.split('_')[-1].lower())
        validate_skel['asf_path'] = validate_skel['path'].map(lambda x: x.parent / (x.parent.stem + '.asf'))
        test_skel= pd.DataFrame({'path': self.test_amc})
        test_skel['Subject'] = test_skel['path'].map(lambda x: x.parent.stem)
        test_skel['Activity'] = test_skel['path'].map(lambda x: x.stem.split('_')[-1].lower())
        test_skel['asf_path'] = test_skel['path'].map(lambda x: x.parent / (x.parent.stem + '.asf'))
        return train_skel, validate_skel, test_skel

    def get_npz_data(self):

        train_smpl = pd.DataFrame({'path': self.train_npz})
        train_smpl['Subject'] = train_smpl['path'].map(lambda x: x.parent.stem)
        validate_smpl = pd.DataFrame({'path': self.validate_npz})
        validate_smpl['Subject'] = validate_smpl['path'].map(lambda x: x.parent.stem)
        test_smpl = pd.DataFrame({'path': self.test_npz})
        test_smpl['Subject'] = test_smpl['path'].map(lambda x: x.parent.stem)
        return train_smpl, validate_smpl, test_smpl

# train_motions = []
# train_joints = []

# for files in train_skel['asf_path'].values:
#     train_joints.append(amc.parse_asf(str(files)))

# for files in train_skel['path'].values:
#     train_motions.append(amc.parse_amc(files))

# # joints_name = ['root','lhipjoint','lfemur','ltibia','lfoot','ltoes','rhipjoint','rfemur','rtibia','rfoot','rtoes',
# #                 'lowerback','upperback','thorax','lowerneck','upperneck','head','lclavicle','lhumerus','lradius','lwrist','lhand','lfingers',
# #                 'lthumb','rclavicle','rhumerus','rradius','rwrist','rhand','rfingers','rthumb']
# bones = []
# motions = []
# skeletons = []

# for i in range(len(train_joints)):
#     for j in range(len(train_motions[i])):
#         train_joints[i]['root'].set_motion(train_motions[i][j])
#         c_joints = train_joints[i]['root'].to_dict()
#         for joint in c_joints.values():
#             for coord in range(3):
#                 bones.append(joint.coordinate[coord, 0])
#         frames = np.array(bones).reshape(-1,31,3)
#     motions.append(frames)
# motions = np.array(motions)







# # for i in range(len(train_motions)):
    

# # motion1 = train_joints[0]['root'].set_motion(train_motions[100])

# print("ciao")