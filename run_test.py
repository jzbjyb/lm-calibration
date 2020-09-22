#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
import tensorflow_datasets as tfds
import t5
from t5.models.mesh_transformer_main import console_entry_point
import dataset


if __name__ == '__main__':

  task = t5.data.TaskRegistry.get('uq_arc_easy')
  ds = task.get_dataset(split='dev', sequence_length={'inputs': 128, 'targets': 32})
  print('\n\n\n')
  print('======= A few preprocessed dev examples =======')
  for ex in tfds.as_numpy(ds.take(5)):
    print(ex)
  print('\n\n\n')

  sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
  sys.exit(console_entry_point())
