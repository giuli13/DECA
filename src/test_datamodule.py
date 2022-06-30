from datamodule import Marker2SkelDataModule
import capspose_flags
from absl import app
from absl import flags
import sys


FLAGS = flags.FLAGS
FLAGS(sys.argv)
dm = Marker2SkelDataModule(FLAGS)


dm.prepare_data()
train_data = dm.train_dataloader()
# test_data = dm.test_dataloader()