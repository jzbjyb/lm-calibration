from typing import Dict, List, Tuple, Set, Union, Callable
import os
import functools
import tensorflow as tf
import numpy as np
import string
from collections import defaultdict
import re
import gin
import xgboost as xgb
import mesh_tensorflow as mtf
from mesh_tensorflow.transformer.transformer import delimited_lm_inputs_mask
import t5
from t5.data.utils import get_default_vocabulary


IND2CHAR = dict(zip(range(14), 'ABCDEFGHIJKLMN'))
CHAR2IND = dict(zip('ABCDEFGHIJKLMN', range(14)))


def trivia_preprocessor(ds):
  def normalize_text(text):
    text = tf.strings.lower(text)
    return text

  def to_inputs_and_targets(ex):
    r = {
      'inputs': normalize_text(ex['question']),
      'targets': normalize_text(ex['answer']),
      'weights': ex['weights']
    }
    if 'ind' in ex:
      r['ind'] = ex['ind']
    return r

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


def qa_dataset_fn_onlycorrect_multi(split: str,
                                    shuffle_files: bool=False,
                                    bucket: str='',
                                    domain: str=None,
                                    format: str='tsv',
                                    num_repeat: int=1):
  if domain:
    file = os.path.join(bucket, domain, split + '.' + format)
  else:
    file = os.path.join(bucket, split + '.' + format)
  ds = tf.data.TextLineDataset(file)
  ds = ds.map(functools.partial(
    tf.io.decode_csv, record_defaults=['', ''], field_delim='\t', use_quote_delim=False),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  def flat_map_fn(question: str, answer: str):
    rec = (question, answer)
    def generator():
      for i in range(num_repeat):
        yield rec
    return tf.data.Dataset.from_generator(generator,
                                          output_types=(tf.string, tf.string),
                                          output_shapes=(tf.TensorShape([]), tf.TensorShape([])))
  def map_fn(question: str, answer: str):
    question = tf.strings.regex_replace(question, '\\\\n', '\n')
    return question, answer, 1.0
  ds = ds.flat_map(lambda *ex: flat_map_fn(*ex))
  ds = ds.map(lambda *ex: dict(zip(['question', 'answer', 'weights'], map_fn(*ex))))
  return ds


def qa_dataset_fn(split: str,
                  shuffle_files: bool=False,
                  bucket: str='',
                  domain: str=None,
                  format: str='tsv',
                  use_neg: bool=False,
                  neg_method: str='weight',
                  only_question: bool=False):
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
    if only_question:
      question = tf.strings.split(question, '\n')[0]
    is_correct = correct == 'True'
    if neg_method == 'weight':
      return ind, question, answer, 1.0 if is_correct else -1.0 / 4
    if neg_method == 'indicator':
      return ind, tf.strings.join([question, ('True:' if is_correct else 'False:')], separator=' '), answer, 1.0
    if neg_method == 'indicator_eval':
      return ind, tf.strings.join([question, 'True'], separator=' '), answer, 1.0
    if neg_method == 'indicator_eval_false':
      return ind, tf.strings.join([question, 'False'], separator=' '), answer, 1.0
    raise NotImplementedError
  ds = ds.map(lambda *ex: dict(zip(['ind', 'question', 'answer', 'weights'], map_fn(*ex))))
  ds = ds.filter(lambda *ex: use_neg or ex[-1] == 'True')
  return ds


def qa_dataset_onlyinput_fn(split: str,
                            shuffle_files: bool=False,
                            bucket: str='',
                            domain: str=None,
                            format: str='tsv',
                            only_question: bool=False):
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
    if only_question:
      question = tf.strings.split(question, '\n')[0]
    return ind, '', question, 1.0
  ds = ds.map(lambda *ex: dict(zip(['ind', 'question', 'answer', 'weights'], map_fn(*ex))))
  return ds


def qa_dataset_fn_ret(split: str,
                      shuffle_files: bool=False,
                      bucket: str='',
                      domain: str=None,
                      format: str='tsv',
                      num_ret: int=5,
                      ret_ind: int=0,
                      ret_method: str='q-prepend',
                      onlyinput: bool=False,
                      only_question: bool=False):
  ret_method = set(ret_method.split('-'))
  if domain:
    file = os.path.join(bucket, domain, split + '.' + format)
  else:
    file = os.path.join(bucket, split + '.' + format)
  ds = tf.data.TextLineDataset(file)
  ds = ds.map(functools.partial(
    tf.io.decode_csv, record_defaults=['', '', '', ''] + [''] * num_ret * 2, field_delim='\t', use_quote_delim=False),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
  def map_fn(ind: str, question: str, answer: str, correct: str, *rets):
    question = tf.strings.regex_replace(question, '\\\\n', '\n')
    if only_question:
      question = tf.strings.split(question, '\n')[0]
    is_correct = correct == 'True'
    qrs, ars = rets[:num_ret], rets[num_ret:]
    if 'q' in ret_method:
      if 'append' in ret_method:
        question = tf.strings.join([question, ' \n ',  qrs[ret_ind]])
      elif 'prepend' in ret_method:
        question = tf.strings.join([qrs[ret_ind], ' \n ', question])
      elif 'vis' in ret_method:
        question = tf.strings.join([question, ' \n ', 'RETRIEVED: ', qrs[ret_ind]])
      else:
        raise NotImplementedError
    if 'a' in ret_method:
      if 'append' in ret_method:
        answer = tf.strings.join([answer, ' \n ', ars[ret_ind]])
      elif 'prepend' in ret_method:
        answer = tf.strings.join([ars[ret_ind], ' \n ', answer])
      else:
        raise NotImplementedError
    if onlyinput:
      return ind, '', question, 1.0 if is_correct else -1.0
    else:
      return ind, question, answer, 1.0 if is_correct else -1.0
  ds = ds.map(lambda *ex: dict(zip(['ind', 'question', 'answer', 'weights'], map_fn(*ex))))
  return ds


def is_int(s):
  try:
    int(s)
    return True
  except ValueError:
    return False


def parse_ind(ind: str, get_input: bool=True):
  inds = ind.split('-')
  if get_input:
    ind = [ind for ind in inds if is_int(ind)][0]
    return ind
  raise NotImplementedError


def no_dup_filter_func(example: Dict):  # only assume two options
  target_texts = example['target_text']
  gold = target_texts[0].lower()
  if gold == '<no answer>':
    return False
  gold = set(re.split(r'\W+', gold)) - {''}
  l = len(target_texts)
  assert l % 2 == 0, 'should only have two answer options'
  other = target_texts[l // 2].lower()
  other = set(re.split(r'\W+', other)) - {''}
  if len(other & gold) > 0:
    keep = False
  else:
    keep = True
  return keep


def get_m_per_n(arr: List, mn: Tuple[int, int]):
  if mn is None:
    return arr
  m, n = mn
  new_arr = []
  for i in range(0, len(arr), n):
    new_arr.extend(arr[i:i+m])
  return new_arr


def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))


def read_score_data(filename: str, mixture: str, split: str,
                    topk: int=None, m_per_n: Tuple[int, int]=None, filter_func: Callable=None, **kwargs):
  if filter_func is None:
    filter_func = lambda x: True
  mix = t5.data.MixtureRegistry.get(mixture)
  ds = mix.get_dataset_in_order(split=split, sequence_length={'inputs': 512, 'targets': 512}, shuffle=False)
  vocab = get_default_vocabulary()

  with open(filename, 'r') as fin:
    if 'inp_perp' in kwargs and kwargs['inp_perp'] is not None:
      inp_perp_fin = open(kwargs['inp_perp'], 'r')
    else:
      inp_perp_fin = None

    prev_inp = None
    prev_ind = None
    prev_task = None
    prev_gold = None
    scores = []
    input_len = []
    target_len = []
    input_tokens = []
    target_tokens = []
    input_texts = []
    target_texts = []
    logprobs = []
    targets = []

    for count_ind, l in enumerate(fin):
      # read dataset
      try:
        task, ex = next(ds)
      except StopIteration:
        break
      # read other features
      if inp_perp_fin:
        inp_perp_ls = inp_perp_fin.readline().strip().split('\t')
        inp_perp = float(inp_perp_ls[0])
        inp_prep_len = len(inp_perp_ls[2].split(',0', 1)[0].split(','))
        inp_perp /= (inp_prep_len or 1)
      else:
        inp_perp = None

      weight = float(ex['weights'].numpy())
      inp = ex['inputs_plaintext'].numpy().decode()
      ind = parse_ind(ex['ind'].numpy().decode())

      ls = l.strip().split('\t')
      if len(ls) == 1:
        score = ls[0]
        score = float(score)
      elif len(ls) in {4, 6, 7}:
        score, inp_tokens, tgt_tokens, logprob = ls[:4]
        score = float(score)
        if 'fast' not in kwargs or not kwargs['fast']:
          tgt_tokens = [int(i) for i in tgt_tokens.split(',0', 1)[0].split(',')]
          logprob = [float(i) for i in logprob.split(',')]
          tgt_tokens = [vocab.decode([i]) for i in tgt_tokens]
          logprob = (task, inp, tgt_tokens, logprob[:len(tgt_tokens)], int(weight == 1))  # inp has '\n'
        else:
          logprob = None
      else:
        raise NotImplementedError
      if prev_ind is not None and (prev_task != task or prev_ind != ind):
        var = np.var(np.exp(np.array(scores)))
        score_var = [var] * len(scores)
        _topk = topk if topk else len(scores)
        yield_result = {'task': prev_task, 'ind': prev_ind,
                        'log_prob': get_m_per_n(scores[:_topk], m_per_n), 'prob_var': get_m_per_n(score_var[:_topk], m_per_n),
                        'input_len': get_m_per_n(input_len[:_topk], m_per_n), 'target_len': get_m_per_n(target_len[:_topk], m_per_n),
                        'target': get_m_per_n(targets[:_topk], m_per_n),
                        'input_tokens': get_m_per_n(input_tokens[:_topk], m_per_n), 'target_tokens': get_m_per_n(target_tokens[:_topk], m_per_n),
                        'input_text': get_m_per_n(input_texts[:_topk], m_per_n), 'target_text': get_m_per_n(target_texts[:_topk], m_per_n),
                        'logprobs': get_m_per_n(logprobs[:_topk], m_per_n)}
        if inp_perp is not None:
          yield_result['inp_perp'] = get_m_per_n(([inp_perp] * len(scores))[:_topk], m_per_n)
        if filter_func(yield_result):
          yield yield_result
        scores = []
        input_len = []
        target_len = []
        input_tokens = []
        target_tokens = []
        input_texts = []
        target_texts = []
        logprobs = []
        targets = []
      scores.append(score)
      input_len.append(len(ex['inputs'].numpy()))
      target_len.append(len(ex['targets'].numpy()))
      input_tokens.append(ex['inputs'].numpy().tolist())
      target_tokens.append(ex['targets'].numpy().tolist())
      input_texts.append(ex['inputs_plaintext'].numpy().decode('utf-8'))
      target_texts.append(ex['targets_plaintext'].numpy().decode('utf-8'))
      logprobs.append(logprob)
      prev_inp = inp
      prev_ind = ind
      prev_task = task
      '''
      if weight == 1:
        prev_gold = ex['targets_plaintext'].numpy().decode('utf-8')
      else:
        cur = ex['targets_plaintext'].numpy().decode('utf-8')
        weight = int(normalize_answer(cur) == normalize_answer(prev_gold))
      '''
      targets.append(int(weight == 1))
    if len(scores) > 0:
      var = np.var(np.exp(np.array(scores)))
      score_var = [var] * len(scores)
      _topk = topk if topk else len(scores)
      yield_result = {'task': prev_task, 'ind': prev_inp,
                      'log_prob': get_m_per_n(scores[:_topk], m_per_n), 'prob_var': get_m_per_n(score_var[:_topk], m_per_n),
                      'input_len': get_m_per_n(input_len[:_topk], m_per_n), 'target_len': get_m_per_n(target_len[:_topk], m_per_n),
                      'target': get_m_per_n(targets[:_topk], m_per_n),
                      'input_tokens': get_m_per_n(input_tokens[:_topk], m_per_n), 'target_tokens': get_m_per_n(target_tokens[:_topk], m_per_n),
                      'input_text': get_m_per_n(input_texts[:_topk], m_per_n), 'target_text': get_m_per_n(target_texts[:_topk], m_per_n),
                      'logprobs': get_m_per_n(logprobs[:_topk], m_per_n)}
      if inp_perp is not None:
        yield_result['inp_perp'] = get_m_per_n(([inp_perp] * len(scores))[:_topk], m_per_n)
      if filter_func(yield_result):
        yield yield_result


def convert_data_to_dmatrix(data, split: float=0.8):
  f1 = [v for d in data['log_prob'] for v in d]
  f2 = [v for d in data['input_len'] for v in d]
  f3 = [v for d in data['target_len'] for v in d]
  f4 = [v for d in data['prob_var'] for v in d]
  fs = [f1, f2, f3, f4]
  if 'inp_perp' in data:
    f5 = [v for d in data['inp_perp'] for v in d]
    fs.append(f5)
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
