from typing import Tuple, Union
import functools
import hashlib
import os
import tensorflow as tf
import t5
import gin
from .utils import trivia_preprocessor

UNIFIEDQA_GS = 'gs://unifiedqa/data'
UNIFIEDQA_PREP_GS = 'gs://neulab-qa/data/unifiedqa'
TRAIN_DOMAINS = [('arc_easy', ('train', 'dev', 'test')),
                 ('ai2_science_elementary', ('train', 'dev', 'test')),
                 ('openbookqa', ('train', 'dev', 'test')),
                 ('qasc', ('train', 'dev')),
                 ('winogrande_l', ('train', 'dev')),
                 ('commonsenseqa', ('train', 'dev'))]
TEST_DOMAINS = [('arc_hard', ('train', 'dev', 'test')),
                ('ai2_science_middle', ('train', 'dev', 'test')),
                ('winogrande_m', ('train', 'dev')),
                ('winogrande_s', ('train', 'dev')),
                ('mctest_corrected_the_separator', ('train', 'dev')),
                ('physical_iqa', ('train', 'dev', 'test')),
                ('social_iqa', ('train', 'dev')),
                ('race_string', ('train', 'dev', 'test'))]
DOMAINS = TRAIN_DOMAINS + TEST_DOMAINS


def compose_qa_pair(question: str,
                    answer: str,
                    is_neg: bool,
                    neg_method: str='weight') -> Tuple[str, str, Union[float, str]]:
  if neg_method == 'weight':
    if is_neg:
      return question, answer, -1.0
    return question, answer, 1.0
  elif neg_method == 'indicator':
    return question + ' ' + ('False:' if is_neg else 'True:'), answer, 1.0
  elif neg_method == 'bool':
    if is_neg:
      return question, answer, 'False'
    return question, answer, 'True'
  else:
    raise NotImplementedError


def one2multi(in_fname: str, out_fname: str):
  mc = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)']
  with tf.io.gfile.GFile(in_fname, 'r') as fin, tf.io.gfile.GFile(out_fname, 'w') as fout:
    for line_ind, line in enumerate(fin):
      question, answer = line.strip().split('\t')
      question = question.strip()
      answer = answer.strip()

      cc = 0
      remain_answer = question.split(mc[0], 1)  # TODO: assume ABCD is not in question-answer pairs.
      remain_answer = None if len(remain_answer) < 2 else remain_answer[1]
      for i in range(len(mc)):
        if remain_answer is None:
          break
        if i < len(mc) - 1:
          _answer = remain_answer.split(mc[i + 1], 1)
        else:
          _answer = [remain_answer]  # remove the context
        remain_answer = None
        if len(_answer) == 2:
          _answer, remain_answer = _answer
        else:
          _answer = _answer[0].split('\\n', 1)[0]
        _answer = _answer.strip()
        #print('"{}" "{}"'.format(_answer, answer))
        cc += int(_answer == answer)
        q, a, b = compose_qa_pair(question, _answer, is_neg=_answer != answer, neg_method='bool')
        fout.write('{}\t{}\t{}\t{}\n'.format(line_ind, q, a, b))
      assert cc >= 1, '#correct answer is {}, should >= 1, question is "{}", answer is "{}"'.format(cc, question, answer)


def unifiedqa_dataset_fn(split: str,
                         shuffle_files: bool=False,
                         domain: str='',
                         format: str='tsv',
                         use_neg: bool=False,
                         neg_method: str='weight'):
  file = os.path.join(UNIFIEDQA_PREP_GS, domain, split + '.' + format)

  ds = tf.data.TextLineDataset(file)
  ds = ds.map(functools.partial(
    tf.io.decode_csv, record_defaults=['', '', '', ''], field_delim='\t', use_quote_delim=False),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  def map_fn(ind: str, question: str, answer: str, correct: str):
    # ind = tf.strings.join([file, ind], separator='|')
    question = tf.strings.regex_replace(question, '\\\\n', '\n')
    is_correct = correct == 'True'
    if neg_method == 'weight':
      return question, answer, 1.0 if is_correct else -1.0 / 4
    if neg_method == 'indicator':
      return tf.strings.join([question, ('True:' if is_correct else 'False:')], separator=' '), answer, 1.0
    raise NotImplementedError
  ds = ds.map(lambda *ex: dict(zip(['question', 'answer', 'weights'], map_fn(*ex))))
  ds = ds.filter(lambda *ex: use_neg or ex[-1] == 'True')
  return ds


@gin.configurable
def build_uq(neg_method: str='indicator'):
  for domain, splits in DOMAINS:
    t5.data.TaskRegistry.add(
      'uq_{}'.format(domain),
      dataset_fn=functools.partial(unifiedqa_dataset_fn, domain=domain, use_neg=True, neg_method=neg_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])

    t5.data.MixtureRegistry.remove('uq_{}_mix'.format(domain))
    t5.data.MixtureRegistry.add('uq_{}_mix'.format(domain), ['uq_{}'.format(domain)], default_rate=1.0)

  t5.data.MixtureRegistry.remove('uq_train_mix')
  t5.data.MixtureRegistry.add('uq_train_mix', ['uq_{}'.format(domain) for domain, _ in TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_test_mix')
  t5.data.MixtureRegistry.add('uq_test_mix', ['uq_{}'.format(domain) for domain, _ in TEST_DOMAINS], default_rate=1.0)
