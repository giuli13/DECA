import c3d
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
import csv
import numpy as np

# a = np.array([1,2,3])
# b = np.array([9,8,7])
# np.savez('/home/giuliamartinelli/Documents/Repositories/DECA/dataset/numpies.npz', a, b)

data = np.load('/home/giuliamartinelli/Documents/Repositories/DECA/dataset/CMU/train.npz',allow_pickle=True)

BASE_DIR_C3D = Path('/media/giuliamartinelli/c73e5e73-45ea-4b43-8dbb-cebfdacf033d/giuliamartinelli') / 'CMU'
BASE_DIR_SMPL = Path('/media/giuliamartinelli/c73e5e73-45ea-4b43-8dbb-cebfdacf033d/giuliamartinelli') / 'CMU' / 'SMPL'
BASE_DIR_SKEL = Path('/media/giuliamartinelli/c73e5e73-45ea-4b43-8dbb-cebfdacf033d/giuliamartinelli') / 'CMU' / 'allasfamc' / 'all_asfamc'

paths = Collect_Paths(BASE_DIR_C3D = BASE_DIR_C3D, BASE_DIR_SKEL=BASE_DIR_SKEL,BASE_DIR_SMPL=BASE_DIR_SMPL)

# train, validation, test = paths.get_c3d_data()
# train_skel, validation_skel, test_skel = paths.get_amc_data()
# train_smpl, validation_smpl, test_smpl = paths.get_npz_data()

train_c3d_path = "/media/giuliamartinelli/c73e5e73-45ea-4b43-8dbb-cebfdacf033d/giuliamartinelli/CMU/allc3d_234/subjects/28/28_02.c3d"
train_amc_path = "/media/giuliamartinelli/c73e5e73-45ea-4b43-8dbb-cebfdacf033d/giuliamartinelli/CMU/allasfamc/all_asfamc/subjects/28/28_02.amc"
train_asf_path = "/media/giuliamartinelli/c73e5e73-45ea-4b43-8dbb-cebfdacf033d/giuliamartinelli/CMU/allasfamc/all_asfamc/subjects/28/28.asf"
train_smpl_path = "/media/giuliamartinelli/c73e5e73-45ea-4b43-8dbb-cebfdacf033d/giuliamartinelli/CMU/SMPL/28/28_02_stageii.npz"
# train_file = natsort.natsorted(train['path'].values)
# train_file_skel_amc = natsort.natsorted(train_skel['path'].values)
# train_file_skel_asf = natsort.natsorted(train_skel['asf_path'].values)
# train_file_smpl = natsort.natsorted(train_smpl['path'].values)


joints = amc.parse_asf(train_asf_path)
motions = amc.parse_amc(train_amc_path)
 
bdata = np.load(str(train_smpl_path))


comp_device = torch.device('cuda')
subject_gender = bdata['gender']

bm_fname = "/home/giuliamartinelli/Documents/Code/smplify-x/models/smplx/SMPLX_NEUTRAL.npz"
bm = BodyModel(bm_fname=bm_fname, num_betas=10).to(comp_device)
faces = c2c(bm.f)

time_length = len(bdata['trans'])

body_parms = {
    'root_orient': torch.Tensor(bdata['root_orient']).to(comp_device), # controls the global root orientation
    'pose_body': torch.Tensor(bdata['pose_body']).to(comp_device), # controls the body
    'pose_hand': torch.Tensor(bdata['pose_hand']).to(comp_device), # controls the finger articulation
    'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
    'betas': torch.Tensor(np.repeat(bdata['betas'][:10][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
}

body_pose_hand = bm(**{k:v for k,v in body_parms.items() if k in ['root_orient','pose_body', 'betas', 'pose_hand','trans']})

n_joints = 31
n_frames = len(motions)

# points3d = np.empty((n_frames, n_joints, 3), np.float32)

# for frame, motion in enumerate(motions):
#     joints['root'].set_motion(motion)
#     for jid, j in enumerate(joints.values()):
#         points3d[frame, jid] = np.squeeze(j.coordinate)

# points3d = points3d * 0.056444 

from ezc3d import c3d
c = c3d(train_c3d_path)
print(c['parameters']['POINT']['USED']['value'][0])
scale = abs(c["parameters"]["POINT"]["SCALE"]["value"][0])
print(scale)
point_data = c['data']['points'][:,:41,:]
fig = plt.figure()
ax = Axes3D(fig)
body_edges = [[0,1],[0,2],[0,3],[1,4],[2,5],[3,6],[4,7],[5,8],[6,9],[7,10],[8,11],[9,12],[12,13],[12,14],
              [12,15],[13,16],[14,17],[16,18],[17,19],[18,20],[19,21],[15,22],[22,23],[22,24],[20,25],[25,26],
              [26,27],[20,28],[28,29],[29,30],[20,31],[31,32],[32,33],[20,34],[34,35],[35,36],[20,37],[37,38],[38,39],
              [21,40],[40,41],[41,42],[21,43],[43,44],[44,45],[21,46],[46,47],[47,48],[21,49],[49,50],[50,51],[21,52],
              [52,53],[53,54]]

def draw_frame(i):
    ax.cla()
    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(0.2, 1.4)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    xs, ys, zs = [], [], []
    # np.array([0, -20, -100], dtype=np.float32)
    for x,y,z in zip(point_data[0],point_data[1],point_data[2]):
      xs.append(x[i]/1000)
      ys.append(y[i]/1000)
      zs.append(z[i]/1000)
      ax.plot(xs, ys, zs, 'g.')
    joints['root'].set_motion(motions[i])
    c_joints = joints['root'].to_dict()
    xskel, yskel, zskel = [], [], []
    for joint in c_joints.values():
      xskel.append(joint.coordinate[0, 0]*0.056444)
      yskel.append(joint.coordinate[1, 0]*0.056444)
      zskel.append(joint.coordinate[2, 0]*0.056444)
    ax.plot(zskel, xskel, yskel, 'b.')

    smpl_joints = c2c(body_pose_hand.Jtr[i])
    xsmpl,ysmpl,zsmpl = [],[],[]
    for smpl_joint in smpl_joints:
      xsmpl.append(smpl_joint[0])
      ysmpl.append(smpl_joint[1])
      zsmpl.append(smpl_joint[2])
    ax.plot(xsmpl, ysmpl, zsmpl, 'r.')

    plt.show()

    for joint in c_joints.values():
      child = joint
      if child.parent is not None:
        parent = child.parent
        xskel = [child.coordinate[0, 0]*0.056444, parent.coordinate[0, 0]*0.056444]
        yskel = [child.coordinate[1, 0]*0.056444, parent.coordinate[1, 0]*0.056444]
        zskel = [child.coordinate[2, 0]*0.056444, parent.coordinate[2, 0]*0.056444]
        ax.plot(zskel, xskel, yskel, 'b')

    for edge in body_edges:
        xsmpl = [smpl_joints[edge[0]][0],smpl_joints[edge[1]][0]]
        ysmpl = [smpl_joints[edge[0]][1],smpl_joints[edge[1]][1]]
        zsmpl = [smpl_joints[edge[0]][2],smpl_joints[edge[1]][2]]
        ax.plot(xsmpl, ysmpl, zsmpl, 'r')
        

out_path = '28_02.gif'
FuncAnimation(fig, draw_frame, range(0, point_data.shape[2], 10)).save(out_path, 
                                                  bitrate=8000,
                                                  fps=8)
plt.close('all')
FileLink(out_path)

print('ciao')