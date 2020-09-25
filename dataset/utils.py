import os
import functools
import tensorflow as tf


IND2CHAR = dict(zip(range(4), 'ABCD'))
CHAR2IND = dict(zip('ABCD', range(4)))


def trivia_preprocessor(ds):
  def normalize_text(text):
    text = tf.strings.lower(text)
    return text

  def to_inputs_and_targets(ex):
    return {
      'inputs': normalize_text(ex['question']),
      'targets': normalize_text(ex['answer']),
      'weights': ex['weights']
    }

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def qa_dataset_fn(split: str,
                  shuffle_files: bool=False,
                  bucket: str='',
                  domain: str=None,
                  format: str='tsv',
                  use_neg: bool=False,
                  neg_method: str='weight'):
  if domain:
    file = os.path.join(bucket, domain, split + '.' + format)
  else:
    file = os.path.join(bucket, split + '.' + format)

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
    if neg_method == 'indicator_eval':
      return tf.strings.join([question, 'True'], separator=' '), answer, 1.0
    raise NotImplementedError
  ds = ds.map(lambda *ex: dict(zip(['question', 'answer', 'weights'], map_fn(*ex))))
  ds = ds.filter(lambda *ex: use_neg or ex[-1] == 'True')
  return ds
