from typing import Dict, List, Tuple
import os
import functools
import tensorflow as tf
import numpy as np
import gin
import xgboost as xgb
import mesh_tensorflow as mtf
from mesh_tensorflow.transformer.transformer import delimited_lm_inputs_mask
import t5


IND2CHAR = dict(zip(range(14), 'ABCDEFGHIJKLMN'))
CHAR2IND = dict(zip('ABCDEFGHIJKLMN', range(14)))


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


def qa_dataset_fn_onlycorrect(split: str,
                              shuffle_files: bool=False,
                              bucket: str='',
                              domain: str=None,
                              format: str='tsv'):
  if domain:
    file = os.path.join(bucket, domain, split + '.' + format)
  else:
    file = os.path.join(bucket, split + '.' + format)
  ds = tf.data.TextLineDataset(file)
  ds = ds.map(functools.partial(
    tf.io.decode_csv, record_defaults=['', ''], field_delim='\t', use_quote_delim=False),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  def map_fn(question: str, answer: str):
    question = tf.strings.regex_replace(question, '\\\\n', '\n')
    return question, answer, 1.0
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
    if neg_method == 'indicator_eval_false':
      return tf.strings.join([question, 'False'], separator=' '), answer, 1.0
    raise NotImplementedError
  ds = ds.map(lambda *ex: dict(zip(['question', 'answer', 'weights'], map_fn(*ex))))
  ds = ds.filter(lambda *ex: use_neg or ex[-1] == 'True')
  return ds


def read_score_data(filename: str, mixture: str, split: str):
  mix = t5.data.MixtureRegistry.get(mixture)
  ds = mix.get_dataset_in_order(split=split, sequence_length={'inputs': 512, 'targets': 512}, shuffle=False)

  with open(filename, 'r') as fin:
    prev_inp = None
    scores = []
    input_len = []
    target_len = []
    targets = []
    for l in fin:
      try:
        ex = next(ds)
      except StopIteration:
        break
      weight = float(ex['weights'].numpy())
      inp = ex['inputs_plaintext'].numpy().decode()
      score = l.strip().split('\t', 1)[0]
      score = float(score)
      if prev_inp is not None and prev_inp != inp:
        var = np.var(np.exp(np.array(scores)))
        score_var = [var] * len(scores)
        yield {'log_prob': scores, 'prob_var': score_var,
               'input_len': input_len, 'target_len': target_len,
               'target': targets}
        scores = []
        input_len = []
        target_len = []
        targets = []
      scores.append(score)
      input_len.append(len(ex['inputs'].numpy()))
      target_len.append(len(ex['targets'].numpy()))
      targets.append(int(weight == 1))
      prev_inp = inp
    if len(scores) > 0:
      var = np.var(np.exp(np.array(scores)))
      score_var = [var] * len(scores)
      yield {'log_prob': scores, 'prob_var': score_var,
             'input_len': input_len, 'target_len': target_len,
             'target': targets}


def convert_data_to_dmatrix(data, split: float=0.8):
  f1 = [v for d in data['log_prob'] for v in d]
  f2 = [v for d in data['input_len'] for v in d]
  f3 = [v for d in data['target_len'] for v in d]
  f4 = [v for d in data['prob_var'] for v in d]
  fs = [f1, f2, f3, f4]
  for f in fs:
    assert len(f) == len(fs[0])
  x = np.array(fs).transpose()
  y = np.array([v for d in data['target'] for v in d])
  if split:
    perm = np.random.permutation(len(x))
    x = x[perm]
    y = y[perm]
    split = int(split * len(x))
    x_train, x_dev = x[:split], x[split:]
    y_train, y_dev = y[:split], y[split:]
    dm_train = xgb.DMatrix(x_train, label=y_train)
    dm_dev = xgb.DMatrix(x_dev, label=y_dev)
    return dm_train, dm_dev
  else:
    return xgb.DMatrix(x, label=y), None


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
    sub_len = tf.reshape(tf.where(tf.equal(x, sep_ind)) + 1, [-1])
    sub_len = sub_len - tf.concat([[0], sub_len[:-1]], axis=0)
    xs = tf.split(x, sub_len, num=num_sep)
    # skip the last one because it will be handled by t5 code
    xs = [_trim(x, max_len=max_len) for x in xs[:-1]] + [_trim(xs[-1], max_len=max_len, eos=False, pad=False)]
    x = tf.concat(xs, axis=0)
    return x

  assert sequence_length[feature_key] % num_sep == 0, 'seq len should be divided by num_sep'
  return ds.map(lambda ex: {k: _concat(ex[k], sequence_length[k] // num_sep) if k == feature_key else ex[k] for k in ex})


@gin.configurable
def mc_softmax_loss_fn(self, context, logits, targets, weights, output_vocab_dim):
  off_value = self.label_smoothing / output_vocab_dim.size
  on_value = 1.0 - self.label_smoothing + off_value
  soft_targets = mtf.one_hot(
    targets,
    output_vocab_dim,
    dtype=context.activation_dtype,
    on_value=on_value,
    off_value=off_value)

  z_loss = self.z_loss if context.train else 0.0

  if soft_targets.dtype.is_integer:
    # hard targets
    if (set(soft_targets.shape.dims) != set(logits.shape.dims).difference([output_vocab_dim])):
      raise ValueError(
          "softmax_cross_entropy_with_logits with hard targets "
          "dims in targets=%s should be dims in logits=%s other than "
          "vocab_dim=%s" % (soft_targets, logits, output_vocab_dim))
    soft_targets = mtf.one_hot(soft_targets, output_vocab_dim, dtype=logits.dtype)
  elif set(soft_targets.shape.dims) != set(logits.shape.dims):
    raise ValueError(
        "softmax_cross_entropy_with_logits with soft targets "
        "dims in targets=%s should be dims in logits=%s"% (soft_targets, logits))
  if output_vocab_dim not in logits.shape.dims:
    raise ValueError("vocab_dim must be in logits.shape.dims")

  print(logits.shape, soft_targets.shape, '*' * 1000)

  log_z = mtf.reduce_logsumexp(logits, output_vocab_dim)
  log_softmax = logits - log_z
  loss = mtf.negative(
      mtf.reduce_sum(log_softmax * soft_targets, reduced_dim=output_vocab_dim))
  if z_loss != 0:
    loss += z_loss * mtf.square(log_z)


  _weights = mtf.layers.weights_nonzero(
    targets, dtype=context.activation_dtype)
  if self.loss_on_targets_only:
    _weights *= mtf.cast(mtf.logical_not(delimited_lm_inputs_mask(targets)),
                         dtype=context.activation_dtype)
  weight_loss = loss * _weights
  if weights is not None:
    weight_loss = weight_loss * mtf.to_bfloat16(weights)
  return (mtf.reduce_sum(weight_loss) /
          self.loss_denominator(targets, context.num_microbatches))
