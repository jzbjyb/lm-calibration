from typing import List, Tuple
import os
import tensorflow as tf
import t5
from .utils import IND2CHAR, trivia_preprocessor

TEST_GS = 'gs://neulab-qa/data/test'


def compose_qa_pair(question: str, choices: List[str], answer: str) -> Tuple[str, str]:
  choices = tf.strings.join([tf.strings.join(['(', IND2CHAR[i], ') ', c])
                             for i, c in enumerate(choices)], separator=' ')
  question = tf.strings.join(['The following are multiple choice questions.', question, choices], separator=' ')
  return question, answer


def test_dataset_fn(split: str, shuffle_files: bool=False):
  if split == 'train':
    split = 'dev'
  csv_dir = os.path.join(TEST_GS, split)
  csv_files = tf.io.gfile.listdir(csv_dir)
  csv_files = [os.path.join(csv_dir, cf) for cf in csv_files]

  ds = tf.data.experimental.CsvDataset(csv_files, record_defaults=['', '', '', '', '', ''])
  ds = ds.map(lambda *ex: dict(zip(['question', 'answer', 'weights'], compose_qa_pair(ex[0], ex[1:5], ex[5]) + (1.0,))))

  return ds


t5.data.TaskRegistry.add(
  'multi_test',
  dataset_fn=test_dataset_fn,
  splits=['train', 'dev', 'val', 'test'],
  text_preprocessor=[trivia_preprocessor],
  postprocess_fn=t5.data.postprocessors.lower_text,
  metric_fns=[t5.evaluation.metrics.accuracy],
  # TODO: add num_input_examples
)

t5.data.MixtureRegistry.remove('multi_test_mix')
t5.data.MixtureRegistry.add('multi_test_mix', ['multi_test'], default_rate=1.0)
