#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import tensorflow_datasets as tfds
import t5
from t5.models.mesh_transformer_main import console_entry_point
from mesh_tensorflow.transformer import utils
import gin
from absl import logging
from dataset import build_uq


if __name__ == '__main__':
  # init gin
  gin.add_config_file_search_path('t5/models/gin')
  utils.parse_gin_defaults_and_flags()

  # build tasks and mixtures
  build_uq()

  # test
  task = t5.data.TaskRegistry.get('uq_arc_easy')
  ds = task.get_dataset(split='dev', sequence_length={'inputs': 128, 'targets': 32})
  print('======= A few preprocessed dev examples =======')
  for ex in tfds.as_numpy(ds.take(5)):
    print(ex)

  # run
  sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
  sys.exit(console_entry_point())
