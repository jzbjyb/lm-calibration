from typing import Tuple, Union
import functools
import os
import tensorflow as tf
import t5
from .utils import trivia_preprocessor

UNIFIEDQA_GS = 'gs://unifiedqa/data'
UNIFIEDQA_PREP_GS = 'gs://neulab-qa/data/unifiedqa'


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
  mc = ['(A)', '(B)', '(C)', '(D)', '(E)']
  with tf.io.gfile.GFile(in_fname, 'r') as fin, tf.io.gfile.GFile(out_fname, 'w') as fout:
    for line in fin:
      question, answer = line.strip().split('\t')

      cc = 0
      remain_answer = question.split(mc[0], 1)  # TODO: assume ABCD is not in question-answer pairs.
      remain_answer = None if len(remain_answer) < 2 else remain_answer[1]
      for i in range(len(mc)):
        if remain_answer is None:
          break
        if i < len(mc) - 1:
          _answer = remain_answer.split(mc[i + 1], 1)
        else:
          _answer = [remain_answer]
        remain_answer = None
        if len(_answer) == 2:
          _answer, remain_answer = _answer
        else:
          _answer = _answer[0]
        _answer = _answer.strip()
        #print('"{}" "{}"'.format(_answer, answer))
        cc += int(_answer == answer)
        q, a, b = compose_qa_pair(question, _answer, is_neg=_answer != answer, neg_method='bool')
        fout.write('{}\t{}\t{}\n'.format(q, a, b))
      assert cc == 1, '#correct answer is {}, not 1'.format(cc)


def unifiedqa_dataset_fn(split: str,
                         shuffle_files: bool=False,
                         domain: str='',
                         format: str='tsv',
                         use_neg: bool=False,
                         neg_method: str='weight'):
  file = os.path.join(UNIFIEDQA_PREP_GS, domain, split + '.' + format)

  ds = tf.data.TextLineDataset(file)
  ds = ds.map(functools.partial(
    tf.io.decode_csv, record_defaults=['', '', ''], field_delim='\t', use_quote_delim=False),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  def map_fn(question, answer, correct):
    is_correct = correct == 'True'
    if neg_method == 'weight':
      return question, answer, 1.0 if is_correct else -1.0
    if neg_method == 'indicator':
      return tf.strings.join([question, ('True:' if is_correct else 'False:')], separator=' '), answer, 1.0
    raise NotImplementedError
  ds = ds.map(lambda *ex: dict(zip(['question', 'answer', 'weights'], map_fn(*ex))))
  ds = ds.filter(lambda *ex: use_neg or ex[-1] == 'True')
  return ds


for domain, splits in [('arc_easy', ('train', 'dev', 'test'))]:
  t5.data.TaskRegistry.add(
    'uq_{}'.format(domain),
    dataset_fn=functools.partial(unifiedqa_dataset_fn, domain=domain, use_neg=True, neg_method='indicator'),
    splits=splits,
    text_preprocessor=[trivia_preprocessor],
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy])

  t5.data.MixtureRegistry.remove('uq_{}_mix'.format(domain))
  t5.data.MixtureRegistry.add('uq_{}_mix'.format(domain), ['uq_{}'.format(domain)], default_rate=1.0)
