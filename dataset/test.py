import os
import csv
import functools
import tensorflow as tf
import t5
import gin
from .utils import IND2CHAR, CHAR2IND, trivia_preprocessor, qa_dataset_fn, qa_dataset_onlyinput_fn, qa_dataset_fn_ret

TEST_GS = 'gs://neulab-qa/data/test'
TEST_PREP_GS = 'gs://neulab-qa/data/test_prep'
TEST_PREP_GS_BT = 'gs://neulab-qa/data/test_prep_bt'
TEST_PREP_GS_BT_REP = 'gs://neulab-qa/data/test_prep_bt_replace'
TEST_PREP_GS_BT_DEDUP = 'gs://neulab-qa/data/test_prep_bt_dedup'
TEST_PREP_GS_BT_DEDUP_REP = 'gs://neulab-qa/data/test_prep_bt_dedup_replace'
TEST_PREP_GS_RET_DRQA = 'gs://neulab-qa/data/test_prep_ret_drqa'
TEST_PREP_GS_RET_DRQA_3S = 'gs://neulab-qa/data/test_prep_ret_drqa_3s'
TEST_PREP_GS_RET_DRQA_3S_BT_REP = 'gs://neulab-qa/data/test_prep_ret_drqa_3s_bt_replace'
TEST_PREP_GS_RET_DRQA_3S_BT_DEDUP_REP = 'gs://neulab-qa/data/test_prep_ret_drqa_3s_bt_dedup_replace'

DOMAINS = [('', ['val', 'dev', 'test'])]

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
def build_test(neg_method: str='indicator', ret_ind: int=0, ret_method: str='q-prepend'):
  t5.data.TaskRegistry.add(
    'test',
    dataset_fn=functools.partial(
        qa_dataset_fn, bucket=TEST_PREP_GS, use_neg=True, neg_method=neg_method),
    splits=DOMAINS[0][1],
    text_preprocessor=[trivia_preprocessor],
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy])
  t5.data.TaskRegistry.add(
    'test_inp',
    dataset_fn=functools.partial(
      qa_dataset_onlyinput_fn, bucket=TEST_PREP_GS),
    splits=DOMAINS[0][1],
    text_preprocessor=[trivia_preprocessor],
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy])
  t5.data.TaskRegistry.add(
    'test_ret_drqa_3s',
    dataset_fn=functools.partial(
      qa_dataset_fn_ret, bucket=TEST_PREP_GS_RET_DRQA_3S, ret_ind=ret_ind, ret_method=ret_method),
    splits=DOMAINS[0][1],
    text_preprocessor=[trivia_preprocessor],
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy])
  t5.data.TaskRegistry.add(
    'test_bt',
    dataset_fn=functools.partial(
      qa_dataset_fn, bucket=TEST_PREP_GS_BT, use_neg=True, neg_method=neg_method),
    splits=DOMAINS[0][1],
    text_preprocessor=[trivia_preprocessor],
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy])
  t5.data.TaskRegistry.add(
    'test_bt_replace',
    dataset_fn=functools.partial(
      qa_dataset_fn, bucket=TEST_PREP_GS_BT_REP, use_neg=True, neg_method=neg_method),
    splits=DOMAINS[0][1],
    text_preprocessor=[trivia_preprocessor],
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy])
  t5.data.TaskRegistry.add(
    'test_bt_dedup_replace',
    dataset_fn=functools.partial(
      qa_dataset_fn, bucket=TEST_PREP_GS_BT_DEDUP_REP, use_neg=True, neg_method=neg_method),
    splits=DOMAINS[0][1],
    text_preprocessor=[trivia_preprocessor],
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy])
  t5.data.TaskRegistry.add(
    'test_ret_drqa_3s_bt_replace',
    dataset_fn=functools.partial(
      qa_dataset_fn_ret, bucket=TEST_PREP_GS_RET_DRQA_3S_BT_REP, ret_ind=ret_ind, ret_method=ret_method),
    splits=DOMAINS[0][1],
    text_preprocessor=[trivia_preprocessor],
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy])
  t5.data.TaskRegistry.add(
    'test_ret_drqa_3s_bt_dedup_replace',
    dataset_fn=functools.partial(
      qa_dataset_fn_ret, bucket=TEST_PREP_GS_RET_DRQA_3S_BT_DEDUP_REP, ret_ind=ret_ind, ret_method=ret_method),
    splits=DOMAINS[0][1],
    text_preprocessor=[trivia_preprocessor],
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy])
  t5.data.TaskRegistry.add(
    'test_ret_drqa_3s_bt_replace_inp',
    dataset_fn=functools.partial(
      qa_dataset_fn_ret, bucket=TEST_PREP_GS_RET_DRQA_3S_BT_REP, ret_ind=ret_ind, ret_method=ret_method, onlyinput=True),
    splits=DOMAINS[0][1],
    text_preprocessor=[trivia_preprocessor],
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy])
  t5.data.TaskRegistry.add(
    'test_ret_drqa_3s_bt_dedup_replace_inp',
    dataset_fn=functools.partial(
      qa_dataset_fn_ret, bucket=TEST_PREP_GS_RET_DRQA_3S_BT_DEDUP_REP, ret_ind=ret_ind, ret_method=ret_method,
      onlyinput=True),
    splits=DOMAINS[0][1],
    text_preprocessor=[trivia_preprocessor],
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy])

  t5.data.MixtureRegistry.remove('test_mix')
  t5.data.MixtureRegistry.add('test_mix', ['test'], default_rate=1.0)
  t5.data.MixtureRegistry.remove('test_inp_mix')
  t5.data.MixtureRegistry.add('test_inp_mix', ['test_inp'], default_rate=1.0)
  t5.data.MixtureRegistry.remove('test_ret_drqa_3s_mix')
  t5.data.MixtureRegistry.add('test_ret_drqa_3s_mix', ['test_ret_drqa_3s'], default_rate=1.0)
  t5.data.MixtureRegistry.remove('test_bt_mix')
  t5.data.MixtureRegistry.add('test_bt_mix', ['test_bt'], default_rate=1.0)
  t5.data.MixtureRegistry.remove('test_bt_replace_mix')
  t5.data.MixtureRegistry.add('test_bt_replace_mix', ['test_bt_replace'], default_rate=1.0)
  t5.data.MixtureRegistry.remove('test_bt_dedup_replace_mix')
  t5.data.MixtureRegistry.add('test_bt_dedup_replace_mix', ['test_bt_dedup_replace'], default_rate=1.0)
  t5.data.MixtureRegistry.remove('test_ret_drqa_3s_bt_replace_mix')
  t5.data.MixtureRegistry.add('test_ret_drqa_3s_bt_replace_mix', ['test_ret_drqa_3s_bt_replace'], default_rate=1.0)
  t5.data.MixtureRegistry.remove('test_ret_drqa_3s_bt_dedup_replace_mix')
  t5.data.MixtureRegistry.add('test_ret_drqa_3s_bt_dedup_replace_mix', ['test_ret_drqa_3s_bt_dedup_replace'], default_rate=1.0)
  t5.data.MixtureRegistry.remove('test_ret_drqa_3s_bt_replace_inp_mix')
  t5.data.MixtureRegistry.add('test_ret_drqa_3s_bt_replace_inp_mix', ['test_ret_drqa_3s_bt_replace_inp'], default_rate=1.0)
  t5.data.MixtureRegistry.remove('test_ret_drqa_3s_bt_dedup_replace_inp_mix')
  t5.data.MixtureRegistry.add('test_ret_drqa_3s_bt_dedup_replace_inp_mix', ['test_ret_drqa_3s_bt_dedup_replace_inp'], default_rate=1.0)
