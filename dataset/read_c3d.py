import c3d
from pathlib import Path
import numpy as np
import natsort
from Collect_Paths import Collect_Paths
import ezc3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import FileLink
from matplotlib.animation import FuncAnimation
import os, sys
sys.path.append('/home/giuliamartinelli/Documents/Repositories/DECA/AMCParser')
import amc_parser as amc

BASE_DIR_C3D = Path('/media/giuliamartinelli/c73e5e73-45ea-4b43-8dbb-cebfdacf033d/giuliamartinelli') / 'CMU'
BASE_DIR_SMPL = Path('/media/giuliamartinelli/c73e5e73-45ea-4b43-8dbb-cebfdacf033d/giuliamartinelli') / 'CMU' / 'SMPL'
BASE_DIR_SKEL = Path('/media/giuliamartinelli/c73e5e73-45ea-4b43-8dbb-cebfdacf033d/giuliamartinelli') / 'CMU' / 'allasfamc' / 'all_asfamc'

paths = Collect_Paths(BASE_DIR_C3D = BASE_DIR_C3D, BASE_DIR_SKEL=BASE_DIR_SKEL,BASE_DIR_SMPL=BASE_DIR_SMPL)

train, validation, test = paths.get_c3d_data()
train_skel, validation_skel, test_skel = paths.get_amc_data()

train_file = natsort.natsorted(train['path'].values)[0]
train_file_skel_amc = natsort.natsorted(train_skel['path'].values)[0]
train_file_skel_asf = natsort.natsorted(train_skel['asf_path'].values)[0]
# for train_file in train['path']:


joints = amc.parse_asf(train_file_skel_asf)
motions = amc.parse_amc(train_file_skel_amc)

from ezc3d import c3d
c = c3d(str(train_file))
print(c['parameters']['POINT']['USED']['value'][0])
scale = abs(c["parameters"]["POINT"]["SCALE"]["value"][0])
print(scale)
point_data = c['data']['points'][:,:41,:]
fig = plt.figure()
ax = Axes3D(fig)

def draw_frame(i):
    ax.cla()
    ax.set_xlim3d(-50, 80)
    ax.set_ylim3d(-20, 40)
    ax.set_zlim3d(-20, 40)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    xs, ys, zs = [], [], []
    for x,y,z in zip(point_data[0],point_data[1],point_data[2]):
      xs.append(x[i]/50)
      ys.append(y[i]/50)
      zs.append(z[i]/50)
      ax.plot(xs, ys, zs, 'g.')
    joints['root'].set_motion(motions[i])
    c_joints = joints['root'].to_dict()
    xskel, yskel, zskel = [], [], []
    for joint in c_joints.values():
      xskel.append(joint.coordinate[0, 0])
      yskel.append(joint.coordinate[1, 0])
      zskel.append(joint.coordinate[2, 0])
    ax.plot(zskel, xskel, yskel, 'b.')

    for joint in c_joints.values():
      child = joint
      if child.parent is not None:
        parent = child.parent
        xskel = [child.coordinate[0, 0], parent.coordinate[0, 0]]
        yskel = [child.coordinate[1, 0], parent.coordinate[1, 0]]
        zskel = [child.coordinate[2, 0], parent.coordinate[2, 0]]
        ax.plot(zskel, xskel, yskel, 'r')

out_path = 'simple_animation_all.gif'
FuncAnimation(fig, draw_frame, range(0, point_data.shape[2], 10)).save(out_path, 
                                                  bitrate=8000,
                                                  fps=8)
plt.close('all')
FileLink(out_path)

print('ciao')