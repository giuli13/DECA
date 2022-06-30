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
import amc_parser as amc
import csv

BASE_DIR_C3D = Path('/media/giuliamartinelli/c73e5e73-45ea-4b43-8dbb-cebfdacf033d/giuliamartinelli') / 'CMU'
BASE_DIR_SMPL = Path('/media/giuliamartinelli/c73e5e73-45ea-4b43-8dbb-cebfdacf033d/giuliamartinelli') / 'CMU' / 'SMPL'
BASE_DIR_SKEL = Path('/media/giuliamartinelli/c73e5e73-45ea-4b43-8dbb-cebfdacf033d/giuliamartinelli') / 'CMU' / 'allasfamc' / 'all_asfamc'

paths = Collect_Paths(BASE_DIR_C3D = BASE_DIR_C3D, BASE_DIR_SKEL=BASE_DIR_SKEL,BASE_DIR_SMPL=BASE_DIR_SMPL)


train, validation, test = paths.get_c3d_data()
train_files = natsort.natsorted(train['path'].values)
validation_files = natsort.natsorted(validation['path'].values)
test_files = natsort.natsorted(test['path'].values)

len_train_c3d = []
train_c3d = []

len_val_c3d = []
val_c3d = []

len_test_c3d = []
test_c3d = []

for i in range(len(train_files)):
    c = c3d(str(train_files[i]))
    name = str(train_files[i])
    train_c3d.append(name)
    len_train_c3d.append(c['data']['points'].shape[-1])

for i in range(len(validation_files)):
    c = c3d(str(validation_files[i]))
    name = str(validation_files[i])
    val_c3d.append(name)
    len_val_c3d.append(c['data']['points'].shape[-1])

for i in range(len(test_files)):
    c = c3d(str(test_files[i]))
    name = str(test_files[i])
    test_c3d.append(name)
    len_test_c3d.append(c['data']['points'].shape[-1])


train_skel, validation_skel, test_skel = paths.get_amc_data()
train_file_skel_amc = natsort.natsorted(train_skel['path'].values)
train_file_skel_asf = natsort.natsorted(train_skel['asf_path'].values)
val_file_skel_amc = natsort.natsorted(validation_skel['path'].values)
val_file_skel_asf = natsort.natsorted(validation_skel['asf_path'].values)
test_file_skel_amc = natsort.natsorted(test_skel['path'].values)
test_file_skel_asf = natsort.natsorted(test_skel['asf_path'].values)

len_train_amc = []
train_amc = []
train_asf = []

len_val_amc = []
val_amc = []
val_asf = []

len_test_amc = []
test_amc = []
test_asf = []

for amc_file in train_file_skel_amc:
  name = str(amc_file)
  len_train_amc.append(len(amc.parse_amc(amc_file)))
  train_amc.append(name)

for asf_file in train_file_skel_asf:
    name = str(asf_file)
    train_asf.append(name)

for amc_file in val_file_skel_amc:
    name = str(amc_file)
    len_val_amc.append(len(amc.parse_amc(amc_file)))
    val_amc.append(name)

for asf_file in val_file_skel_asf:
    name = str(asf_file)
    val_asf.append(name)

for amc_file in test_file_skel_amc:
    name = str(amc_file)
    len_test_amc.append(len(amc.parse_amc(amc_file)))
    test_amc.append(name)

for asf_file in test_file_skel_asf:
    name = str(asf_file)
    test_asf.append(name)

train_smpl, validation_smpl, test_smpl = paths.get_npz_data()
train_file_smpl = natsort.natsorted(train_smpl['path'].values)
val_file_smpl = natsort.natsorted(validation_smpl['path'].values)
test_file_smpl = natsort.natsorted(test_smpl['path'].values)

len_train_smpl = []
train_smpl = []

len_val_smpl = []
val_smpl = []

len_test_smpl = []
test_smpl = []

for smpl_file in train_file_smpl:
    bdata = np.load(str(smpl_file))
    name = str(smpl_file)
    len_train_smpl.append(len(bdata['trans']))
    train_smpl.append(name)

for smpl_file in val_file_smpl:
    bdata = np.load(str(smpl_file))
    name = str(smpl_file)
    len_val_smpl.append(len(bdata['trans']))
    val_smpl.append(name)

for smpl_file in test_file_smpl:
    bdata = np.load(str(smpl_file))
    name = str(smpl_file)
    len_test_smpl.append(len(bdata['trans']))
    test_smpl.append(name)


with open('/home/giuliamartinelli/Documents/Repositories/DECA/dataset/train_data.csv', 'w', newline='') as file:
  writer = csv.writer(file)
  writer.writerow(["File c3d","Len c3d", "File amc", "Len amc", "File asf", "File smpl", "Len smpl"])
  for i in range(len(train_amc)):
    writer.writerow([train_c3d[i],len_train_c3d[i],train_amc[i],len_train_amc[i],train_asf[i],train_smpl[i],len_train_smpl[i]])

with open('/home/giuliamartinelli/Documents/Repositories/DECA/dataset/validation_data.csv', 'w', newline='') as file:
  writer = csv.writer(file)
  writer.writerow(["File c3d","Len c3d", "File amc", "Len amc", "File asf", "File smpl", "Len smpl"])
  for i in range(len(val_amc)):
    writer.writerow([val_c3d[i],len_val_c3d[i],val_amc[i],len_val_amc[i],val_asf[i],val_smpl[i],len_val_smpl[i]])

with open('/home/giuliamartinelli/Documents/Repositories/DECA/dataset/test_data.csv', 'w', newline='') as file:
  writer = csv.writer(file)
  writer.writerow(["File c3d","Len c3d", "File amc", "Len amc", "File asf", "File smpl", "Len smpl"])
  for i in range(len(test_amc)):
    writer.writerow([test_c3d[i],len_test_c3d[i],test_amc[i],len_test_amc[i],test_asf[i],test_smpl[i],len_test_smpl[i]])