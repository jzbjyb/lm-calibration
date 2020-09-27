from typing import Dict, List, Tuple
import os
import functools
import tensorflow as tf
import gin


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


def qa_dataset_fn_oneline(split: str,
                          shuffle_files: bool=False,
                          bucket: str='',
                          domain: str=None,
                          format: str='tsv',
                          num_sep: int=10,
                          sep: str = '|'):
  if domain:
    file = os.path.join(bucket, domain, split + '.' + format)
  else:
    file = os.path.join(bucket, split + '.' + format)
  ds = tf.data.TextLineDataset(file)
  ds = ds.map(functools.partial(
    tf.io.decode_csv, record_defaults=['', ''] + [''] * 2 * num_sep, field_delim='\t', use_quote_delim=False),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  def map_fn(ind: str, question: str, *answer_ind_pairs: List[str]):
    assert len(answer_ind_pairs) % 2 == 0, 'answers should always have correctness'
    question = tf.strings.regex_replace(question, '\\\\n', '\n')
    ans = []
    for i in range(len(answer_ind_pairs) // 2):
      an, ind = answer_ind_pairs[i * 2], answer_ind_pairs[i * 2 + 1]
      ans.append(tf.strings.regex_replace(an, '\\' + sep if sep in {'|'} else sep, ' '))
    assert len(ans) == num_sep, 'should have more answers'
    ans = ans + ['']  # make sure the last one is "sep"
    return question, tf.strings.join(ans, separator=' {} '.format(sep)), 1.0
  ds = ds.map(lambda *ex: dict(zip(['question', 'answer', 'weights'], map_fn(*ex))))
  return ds


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


@gin.configurable
def concat_preprocessor(ds,
                        sequence_length: Dict,
                        feature_key: str='targets',
                        num_sep: int=10,
                        sep: str = '|',
                        **unused_kwargs):
  # TODO: add vocab
  sep_ind = 1820
  def _trim(x, max_len: int, eos: bool=True, pad: bool=True):
    x = x[:-1]  # remove sep
    if eos:
      x = x[:max_len - 1]
      x = tf.concat([x, [1]], axis=0)
    else:
      x = x[:max_len]
    if pad:
      x = tf.pad(x, [[0, max_len - tf.cast(tf.shape(x)[0], tf.int64)]])
    return x

  def _concat(x, max_len):
    print(max_len)
    sub_len = tf.reshape(tf.where(x == sep_ind) + 1, [-1])
    sub_len = sub_len - tf.concat([[0], sub_len[:-1]], axis=0)
    xs = tf.split(x, sub_len, num=num_sep)
    # skip the last one because it will be handled by t5 code
    xs = [_trim(x, max_len=max_len) for x in xs[:-1]] + [_trim(xs[-1], max_len=max_len, eos=False, pad=False)]
    x = tf.concat(xs, axis=0)
    return x

  assert sequence_length[feature_key] % num_sep == 0, 'seq len should be divided by num_sep'
  return ds.map(lambda ex: {k: _concat(ex[k], sequence_length[k] // num_sep) if k == feature_key else ex[k] for k in ex})
