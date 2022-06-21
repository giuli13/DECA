from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import c3d
import os, sys
sys.path.append('/home/giuliamartinelli/Documents/Repositories/DECA/AMCParser')
import amc_parser as amc





BASE_DIR = Path('/media/giuliamartinelli/c73e5e73-45ea-4b43-8dbb-cebfdacf033d/giuliamartinelli') / 'CMU' / 'allasfamc' / 'all_asfamc'
datasets_df = pd.DataFrame({'path': list(BASE_DIR.glob('subjects/*/*.amc'))})
datasets_df['Subject'] = datasets_df['path'].map(lambda x: x.parent.stem)
datasets_df['Activity'] = datasets_df['path'].map(lambda x: x.stem.split('_')[-1].lower())
datasets_df['asf_path'] = datasets_df['path'].map(lambda x: x.parent / (x.parent.stem + '.asf'))



BASE_DIR_C3D = Path('/media/giuliamartinelli/c73e5e73-45ea-4b43-8dbb-cebfdacf033d/giuliamartinelli') / 'CMU'
datasets_motion = pd.DataFrame({'path': list(BASE_DIR_C3D.glob('*/subjects/*/*.c3d'))})
datasets_motion['Subject'] = datasets_motion['path'].map(lambda x: x.parent.stem)
datasets_motion['Activity'] = datasets_motion['path'].map(lambda x: x.stem.split('_')[-1].lower())

datasets_df.sample(3)

test_rec = datasets_motion.iloc[0]
print(test_rec)
test_file = test_rec['path'] 



# joints = amc.parse_asf(test_rec['asf_path'])
# motions = amc.parse_amc(test_rec['path'])

with test_file.open('rb') as hf:
    all_fields = []
    reader = c3d.Reader(hf)
    scale_xyz = np.abs(reader.point_scale) # don't flip everything
    print("ciao")
#     for frame_no, points, _ in reader.read_frames(copy=False):
#         for (x, y, z, err, cam), label in zip(points, 
#                                      reader.point_labels):
#             c_field = {'frame': frame_no, 
#                        'time': frame_no / reader.point_rate,
#                        'point_label': label.strip()}
#             c_field['x'] = scale_xyz*x
#             c_field['y'] = scale_xyz*y
#             c_field['z'] = scale_xyz*z
#             c_field['err'] = err<0
#             c_field['cam'] = cam<0
#             all_fields += [c_field]
# all_df = pd.DataFrame(all_fields)[['time', 'point_label', 'x', 'y', 'z',  'cam', 'err', 'frame']]
# print(all_df)

# from IPython.display import FileLink
# from matplotlib.animation import FuncAnimation
# fig = plt.figure()
# ax = Axes3D(fig)

# def draw_frame(i):
#     ax.cla()
#     ax.set_xlim3d(-50, 10)
#     ax.set_ylim3d(-20, 40)
#     ax.set_zlim3d(-20, 40)
#     joints['root'].set_motion(motions[i])
#     c_joints = joints['root'].to_dict()
#     xs, ys, zs = [], [], []
#     for joint in c_joints.values():
#       xs.append(joint.coordinate[0, 0])
#       ys.append(joint.coordinate[1, 0])
#       zs.append(joint.coordinate[2, 0])
#     ax.plot(zs, xs, ys, 'b.')

#     for joint in c_joints.values():
#       child = joint
#       if child.parent is not None:
#         parent = child.parent
#         xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
#         ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
#         zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
#         ax.plot(zs, xs, ys, 'r')
# out_path = 'simple_animation.gif'
# FuncAnimation(fig, draw_frame, range(0, len(motions), 10)).save(out_path, 
#                                                   bitrate=8000,
#                                                   fps=8)
# plt.close('all')
# FileLink(out_path)








