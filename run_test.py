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
from dataset import build


if __name__ == '__main__':
  # init gin
  gin.add_config_file_search_path('t5/models/gin')
  utils.parse_gin_defaults_and_flags()

  # build tasks and mixtures
  build()

  # test
  mix = t5.data.MixtureRegistry.get('uq_ext_dup_mix')
  ds = mix.get_dataset_in_order(split='dev', sequence_length={'inputs': 512, 'targets': 128}, shuffle=False)
  print('======= A few preprocessed dev examples =======')
  for i, ex in enumerate(ds):
    if i >= 5:
      break
    print(ex)

  # run
  sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
  sys.exit(console_entry_point())
