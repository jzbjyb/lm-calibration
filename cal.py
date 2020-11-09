from typing import List
import argparse
import os
from operator import itemgetter
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from scipy.stats import entropy
import xgboost as xgb
from dataset import build, read_score_data, convert_data_to_dmatrix


def choose_score(scores_li: List[List[float]], weights: List[float]):
  if len(scores_li) == 1:
    return scores_li[0]
  for scores in scores_li:
    if weights[np.argmax(scores)] == 1:
      return scores
  return scores_li[0]


def compute_acc(acc_li: List[float], task_li: List[str], method: str='micro'):
  overall = []
  current = []
  prev_task = None
  for acc, task in zip(acc_li, task_li):
    if prev_task is not None and task != prev_task:
      overall.append(np.mean(current))
      current = []
    current.append(acc)
    if method == 'macro':
      prev_task = task
    elif method == 'micro':
      pass
    else:
      raise NotImplementedError
  if len(current) > 0:
    overall.append(np.mean(current))
  return np.mean(overall)


def get_ece(acc_li: List[float], conf_li: List[float], plot: bool=False):
  num_bins = 20
  margin = 1 / num_bins
  xind = [margin * (i + 0.5) for i in range(num_bins)]

  bins = [[] for _ in range(num_bins)]
  for acc, conf in zip(acc_li, conf_li):
    assert conf >= 0 and conf <= 1, 'confidence {} out of range'.format(conf)
    ind = min(int(conf / margin), num_bins - 1)
    bins[ind].append((conf, acc))

  eces = [(len(bin), np.mean(list(map(itemgetter(0), bin))), np.mean(list(map(itemgetter(1), bin)))) for bin in bins]
  ece, total = 0, 0
  for c, conf, acc in eces:
    if c <= 0:
      continue
    ece += c * np.abs(conf - acc)
    total += c
  ece /= total

  if plot:
    plt.bar(xind, [np.mean(list(map(itemgetter(1), bin))) for bin in bins], margin)
    plt.title('ece {:.3f}'.format(ece))
    plt.ylabel('accuracy')
    plt.xlabel('confidence')
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    plt.plot([0, 1], color='red')
    plt.savefig('test.png')
    plt.close()

  return ece


def compute_ece(acc_li: List[float], conf_li: List[float], task_li: List[str], method: str='micro'):
  overall = []
  current_acc_li = []
  current_conf_li = []
  prev_task = None
  for acc, conf, task in zip(acc_li, conf_li, task_li):
    if prev_task is not None and task != prev_task:
      overall.append(get_ece(current_acc_li, current_conf_li))
      print(prev_task, overall[-1])
      current_acc_li = []
      current_conf_li = []
    current_acc_li.append(acc)
    current_conf_li.append(conf)
    if method == 'macro':
      prev_task = task
    elif method == 'micro':
      pass
    else:
      raise NotImplementedError
  if len(current_acc_li) > 0:
    overall.append(get_ece(current_acc_li, current_conf_li))
    print(task, overall[-1])
  return np.mean(overall)


def acc(mixture: str, score_files: List[str], split: str='dev', num_bt: int=1,
        temp: float=1.0, norm: str = 'softmax', xgb_model_path=None, ana: bool=False, method: str='micro', **kwargs):
  real_acc_li = []
  real_task_li = []
  acc_li = []
  conf_li = []
  task_li = []

  input_len_li = []
  target_len_li = []
  input_tokens_li = []
  target_tokens_li = []
  overlap_ratio_li = []
  logprobs_li = []

  if xgb_model_path:
    xgb_model = xgb.Booster()
    xgb_model.load_model(xgb_model_path)  # load data
  else:
    xgb_model = None

  sample_gens = [read_score_data(sf, mixture, split, **kwargs) for sf in score_files]
  while True:
    try:
      samples = [next(sg) for sg in sample_gens]
    except StopIteration:
      break

    sample = samples[0]
    task = sample['task']
    weights = sample['target']
    scores = choose_score([s['log_prob'] for s in samples], weights)

    if ana:
      input_len_li.extend(sample['input_len'])
      target_len_li.extend(sample['target_len'])
      input_tokens_li.extend(sample['input_tokens'])
      target_tokens_li.extend(sample['target_tokens'])
      for inp, tar in zip(sample['input_tokens'], sample['target_tokens']):
        inp = set(inp)
        tar = set(tar)
        overlap_ratio_li.append(len(inp & tar) / min(len(inp), len(tar)))
      logprobs_li.extend(sample['logprobs'])

    if xgb_model:
      dtest = convert_data_to_dmatrix({k: np.array([v]) for k, v in sample.items()}, split=0)[0]
      scores = np.log(xgb_model.predict(dtest))  # TODO: ntree_limit=xgb_model.best_ntree_limit

    raw_scores = np.array(scores)
    if norm == 'softmax':
      scores = softmax(np.array(scores) / temp)
    elif norm == 'no':
      scores = np.exp(np.array(scores))
    elif norm == 'margin':
      scores = softmax(np.array(scores))
      argind = np.argsort(np.argsort(scores))
      scores_ranked = sorted(scores)
      margin = np.array(scores_ranked) - np.array([0] + scores_ranked[:-1])
      scores = margin[argind]
    elif norm == 'margin_order':
      scores = softmax(np.array(scores))
      argind = np.argsort(np.argsort(scores))
      scores_ranked = sorted(scores)
      margin = np.array(scores_ranked) - np.array([0] + scores_ranked[:-1])
      margin_order = margin * np.array((1.0 - np.array(scores_ranked))[1:].tolist() + [1.0])
      scores = margin_order[argind]
    else:
      raise NotImplementedError
    assert len(scores) == len(weights) and len(scores) % num_bt == 0, 'wrong correspondence'
    # use sum of log prob
    _scores = [np.sum(scores[k * num_bt:k * num_bt + num_bt]) for k in range(len(scores) // num_bt)]
    _raw_scores = [np.sum(raw_scores[k * num_bt:k * num_bt + num_bt]) for k in range(len(raw_scores) // num_bt)]
    _weights = [weights[k * num_bt:k * num_bt + num_bt] for k in range(len(weights) // num_bt)]
    for score, weight in zip(_scores, _weights):
      assert len(np.unique(weight)) == 1, 'wrong correspondence'
      weight = weight[0]
      acc_li.append(weight == 1)
      conf_li.append(score)
      task_li.append(task)
    choice = np.argmax(_scores)
    real_acc_li.append(int(_weights[choice][0] == 1))
    real_task_li.append(task)

  real_acc = compute_acc(real_acc_li, real_task_li, method=method)
  print('acc', real_acc)

  ece = compute_ece(acc_li, conf_li, task_li, method=method)
  print('ece', ece)

  if ana:
    analysis({
      'conf': np.array(conf_li),
      'acc': np.array(acc_li),
      'input_len': np.array(input_len_li),
      'target_len': np.array(target_len_li),
      'input_tokens': np.array(input_tokens_li),
      'target_tokens': np.array(target_tokens_li),
      'overlap_ratio': np.array(overlap_ratio_li),
      'logprobs': np.array(logprobs_li),
    }, 'output/ana', topk=500)


_norm = matplotlib.colors.Normalize(vmin=-70, vmax=0)
_map = cm.ScalarMappable(norm=_norm, cmap=cm.hot)
def to_stype(scalar, correct: bool) -> str:
  return "style='background-color:rgba({:.0f}%, {:.0f}%, {:.0f}%, {:.0f}%); {}' title='{}'".format(
    *([n * 100 for n in _map.to_rgba(scalar)] + (['font-weight:bold' if correct else '']) + [scalar]))


def analysis(data, output, topk=100):
  os.makedirs(output, exist_ok=True)
  conf = data['conf']
  acc = data['acc']
  gap = conf - acc
  plt.hist(gap, bins=20)
  plt.savefig(os.path.join(output, 'gap.png'))
  plt.close()

  under = np.argsort(gap)[:topk]
  over = np.argsort(-gap)[:topk]
  close = np.argsort(np.abs(gap))[:topk]

  for ind, ind_name in [(under, 'under'), (over, 'over'), (close, 'close')]:
    for metric, xmin, xmax in [('input_len', 0, 512), ('target_len', 0, 128), ('input_tokens', 1000, 30000), ('target_tokens', 1000, 30000), ('logprobs', 0, 0)]:
      x = data[metric][ind]
      if metric == 'logprobs':
        with open(os.path.join(output, '{}-{}.html'.format(metric, ind_name)), 'w') as fout:
          for (inp, tgt, lps), _conf, _acc in zip(x, conf[ind], acc[ind]):
            fout.write('<div><div>{}</div>{}</div><hr/>\n'.format(
              inp, '<div>{} {}</div>'.format(
                _conf, ' '.join(['<span {}>{}</span>'.format(to_stype(l, _acc == 1), t) for t, l in zip(tgt, lps)]))))
        continue
      elif '_tokens' in metric:
        x = np.array([t for ts in x for t in ts])
        bins = 100
        plt.hist(x, bins=bins, weights=np.ones(len(x)) / len(x) * 10)
      else:
        bins = 20
        plt.hist(x, bins=bins, weights=np.ones(len(x)) / len(x))
      plt.xlim(xmin, xmax)
      plt.title('avg {}'.format(np.mean(x)))
      plt.savefig(os.path.join(output, '{}-{}.png'.format(metric, ind_name)))
      plt.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='calibration computation')
  parser.add_argument('--method', type=str, help='metric method', default='micro', choices=['micro', 'macro'])
  parser.add_argument('--mix', type=str, help='mixture', default='uq_sub_test_mix')
  parser.add_argument('--split', type=str, help='split', default='dev')
  parser.add_argument('--score', type=str, help='score file', nargs='+')
  parser.add_argument('--inp_perp', type=str, help='feature of input perplexity', default=None)
  parser.add_argument('--num_bt', type=int, help='number of translations per example', default=1)
  parser.add_argument('--temp', type=float, help='temperature of softmax', default=1.0)
  parser.add_argument('--xgb', type=str, help='xgb model path', default=None)
  parser.add_argument('--norm', type=str, help='normalization method', default='softmax', choices=['softmax', 'no', 'margin', 'margin_order'])
  parser.add_argument('--ana', action='store_true')
  args = parser.parse_args()

  # build tasks and mixtures
  build(neg_method='weight')

  acc(args.mix, args.score, args.split, args.num_bt, args.temp, norm=args.norm, xgb_model_path=args.xgb,
      ana=args.ana, method=args.method, inp_perp=args.inp_perp)
