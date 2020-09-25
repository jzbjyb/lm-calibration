import os
import csv
import functools
import tensorflow as tf
import t5
import gin
from .utils import IND2CHAR, CHAR2IND, trivia_preprocessor, qa_dataset_fn

TEST_GS = 'gs://neulab-qa/data/test'
TEST_PREP_GS = 'gs://neulab-qa/data/test_prep'


def one2multi():
  for split in ['dev', 'val', 'test']:
    print(split)
    csv_dir = os.path.join(TEST_GS, split)
    to_csv_file = os.path.join(TEST_PREP_GS, split + '.tsv')
    with tf.io.gfile.GFile(to_csv_file, 'w') as fout:
      files = tf.io.gfile.listdir(csv_dir)
      csv_files = [os.path.join(csv_dir, cf) for cf in files]
      for csv_file in csv_files:
        with tf.io.gfile.GFile(csv_file, 'r') as fin:
          csv_reader = csv.reader(fin)
          csv_writer = csv.writer(fout, delimiter='\t')
          for row in csv_reader:
            topic = csv_file.rsplit('/', 1)[1].rsplit('_', 1)[0]
            ques = row[0].replace('\n', '\\n')
            ans = [a.replace('\n', '\\n') for a in row[1:5]]
            real_ind = CHAR2IND[row[5]]
            choices = ' '.join(['({}) {}'.format(IND2CHAR[i], c) for i, c in enumerate(ans)])
            question = ' '.join([ques, choices])
            assert real_ind >= 0 and real_ind <= 3
            for i, a in enumerate(ans):
              csv_writer.writerow([topic.replace('\t', ' '),
                                   question.replace('\t', ' '),
                                   a.replace('\t', ' '),
                                   'True' if real_ind == i else 'False'])


@gin.configurable
def build_test(neg_method: str='indicator'):
  t5.data.TaskRegistry.add(
    'test',
    dataset_fn=functools.partial(
        qa_dataset_fn, bucket=TEST_PREP_GS, use_neg=True, neg_method=neg_method),
    splits=['train', 'dev', 'val', 'test'],
    text_preprocessor=[trivia_preprocessor],
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy])

  t5.data.MixtureRegistry.remove('test_mix')
  t5.data.MixtureRegistry.add('test_mix', ['test'], default_rate=1.0)
