from typing import List, Dict, Union, Tuple
import argparse
import os
from tqdm import tqdm
from operator import itemgetter
import numpy as np
from scipy.special import softmax
import itertools
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from scipy.stats import entropy
import xgboost as xgb
from dataset import build, read_score_data, convert_data_to_dmatrix
from dataset.utils import no_dup_filter_func
plt.rcParams.update({'font.size': 26})


def choose_score(scores_li: List[List[float]], weights: List[float]):
  if len(scores_li) == 1:
    return scores_li[0]
  for scores in scores_li:
    if weights[np.argmax(scores)] == 1:
      return scores
  return scores_li[0]


def compute_avg(acc_li: List[float], task_li: List[str], method: str='micro', tag: str='acc'):
  overall = []
  current = []
  prev_task = None
  for acc, task in zip(acc_li, task_li):
    if prev_task is not None and task != prev_task:
      overall.append(np.mean(current))
      print('task {} {}: {}'.format(tag, prev_task, overall[-1]))
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
    print('task {} {}: {}'.format(tag, prev_task, overall[-1]))
  return np.mean(overall)


def get_ece(acc_li: List[float], conf_li: List[float], plot: bool=False):
  num_bins = 20
  margin = 1 / num_bins
  xind = [margin * (i + 0.5) for i in range(num_bins)]

  bins = [[] for _ in range(num_bins)]
  for acc, conf in zip(acc_li, conf_li):
    conf = max(0, min(1, conf))
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
    #plt.title('ece {:.3f}'.format(ece))
    plt.ylabel('accuracy')
    plt.xlabel('confidence')
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    plt.tight_layout()
    plt.plot([0, 1], color='red')
    plt.savefig('reliability.pdf')
    plt.close()

    plt.hist(conf_li, bins=num_bins, weights=np.ones(len(conf_li)) / len(conf_li))
    # plt.title('ece {:.3f}'.format(ece))
    plt.ylabel('ratio')
    plt.xlabel('confidence')
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig('dist.pdf')
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
      print('task ece {}: {}'.format(prev_task, overall[-1]))
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
    print('task ece {}: {}'.format(prev_task, overall[-1]))
  get_ece(acc_li, conf_li, plot=True)
  return np.mean(overall)


def concat_paraphrase(paras: List[Tuple[str, str, List, List]]):
  tasks = None
  inps = None
  tgts = []
  logprobs = []
  for i, (task, inp, tgt, logprob, weight) in enumerate(paras):
    if i == 0:
      tasks = task
      inps = inp
    tgts.extend(tgt)
    logprobs.extend(logprob)
    tgts.append(('({:.3f})' + '&nbsp;' * 6).format(np.exp(np.sum(logprob))))
    logprobs.append(0.0)
  return tasks, inps, tgts, logprobs, weight


def acc(mixture: str, score_files: List[str], split: str='dev', num_bt: int=1,
        temp: float=1.0, norm: str='softmax', xgb_model_path=None, ana: bool=False,
        method: str='micro', topk: int=None, m_per_n: Tuple[int, int]=None, **kwargs):
  real_acc_li = []
  real_task_li = []
  real_entropy_li = []
  acc_li = []
  conf_li = []
  raw_prob_li = []
  task_li = []

  input_len_li = []
  target_len_li = []
  input_tokens_li = []
  target_tokens_li = []
  overlap_ratio_li = []
  logprobs_li = []
  ind_li = []

  if xgb_model_path:
    xgb_model = xgb.Booster()
    xgb_model.load_model(xgb_model_path)  # load data
  else:
    xgb_model = None

  sample_gens = [read_score_data(sf, mixture, split, topk=topk, m_per_n=m_per_n, filter_func=None, **kwargs) for sf in score_files]
  while True:
    try:
      samples = [next(sg) for sg in sample_gens]
    except StopIteration:
      break

    sample = samples[0]
    task = sample['task']
    ind = sample['ind']
    weights = sample['target']
    scores = choose_score([s['log_prob'] for s in samples], weights)

    if ana:
      if num_bt > 1:
        sample['input_len'] = [sample['input_len'][i] for i in range(0, len(sample['input_len']), num_bt)]
        sample['target_len'] = [np.sum(sample['target_len'][i:i+num_bt]) for i in range(0, len(sample['target_len']), num_bt)]
        sample['input_tokens'] = [sample['input_tokens'][i] for i in range(0, len(sample['input_tokens']), num_bt)]
        sample['target_tokens'] = [list(itertools.chain(*sample['target_tokens'][i:i+num_bt])) for i in range(0, len(sample['target_tokens']), num_bt)]
        sample['logprobs'] = [concat_paraphrase(sample['logprobs'][i:i + num_bt]) for i in range(0, len(sample['logprobs']), num_bt)]

      input_len_li.extend(sample['input_len'])
      target_len_li.extend(sample['target_len'])
      input_tokens_li.extend(sample['input_tokens'])
      target_tokens_li.extend(sample['target_tokens'])
      ind_li.extend(['{}-{}'.format(task, ind)] * len(sample['input_len']))
      for inp, tar in zip(sample['input_tokens'], sample['target_tokens']):
        inp = set(inp)
        tar = set(tar)
        overlap_ratio_li.append(len(inp & tar) / min(len(inp), len(tar)))
      logprobs_li.extend(sample['logprobs'])

    if xgb_model:
      dtest = convert_data_to_dmatrix({k: np.array([v]) for k, v in sample.items()}, split=0)[0]
      scores = np.log(xgb_model.predict(dtest))  # TODO: ntree_limit=xgb_model.best_ntree_limit

    raw_scores = np.exp(np.array(scores))
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
    elif norm == 'margin_order_nosm':
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

    if num_bt > 1 and norm == 'no':
      _scores = [s / num_bt for s in _scores]
      _scores = softmax(np.array(_scores) / temp)

    for score, rscore, weight in zip(_scores, _raw_scores, _weights):
      assert len(np.unique(weight)) == 1, 'wrong correspondence'
      weight = weight[0]
      acc_li.append(weight == 1)
      conf_li.append(score)
      task_li.append(task)
      raw_prob_li.append(rscore)
    choice = np.argmax(_scores)
    real_acc_li.append(int(_weights[choice][0] == 1))
    real_entropy_li.append(entropy(_scores, base=len(_scores)))
    real_task_li.append(task)

  real_acc = compute_avg(real_acc_li, real_task_li, method=method, tag='acc')
  print('count', len(real_acc_li))
  print('acc', real_acc)

  ece = compute_ece(acc_li, conf_li, task_li, method=method)
  print('ece', ece)

  ent = compute_avg(real_entropy_li, real_task_li, method=method, tag='entropy')
  print('entropy', ent)

  if ana:
    return {
      'ind': np.array(ind_li),
      'conf': np.array(conf_li),
      'raw_prob': np.array(raw_prob_li),
      'acc': np.array(acc_li),
      'input_len': np.array(input_len_li),
      'target_len': np.array(target_len_li),
      'input_tokens': np.array(input_tokens_li),
      'target_tokens': np.array(target_tokens_li),
      'overlap_ratio': np.array(overlap_ratio_li),
      'logprobs': np.array(logprobs_li),
    }


def agg_ana_data(data: Dict, key: str, other_li: List):
  prev_ind = None
  new_li = []
  new_ind_li = []
  cur_collect = []
  for ind, cur, other in zip(data['ind'], data[key], other_li):
    if prev_ind is not None and ind != prev_ind:
      new_li.extend([cur_collect] * len(cur_collect))
      new_ind_li.extend(list(range(len(cur_collect))))
      cur_collect = []
    if type(cur) is list:
      cur_collect.append(cur + [other])
    elif type(cur) is np.ndarray:
      cur_collect.append(cur.tolist() + [other])
    elif type(cur) is tuple:
      cur_collect.append(cur + (other,))
    else:
      raise ValueError
    prev_ind = ind
  if len(cur_collect) > 0:
    new_li.extend([cur_collect] * len(cur_collect))
    new_ind_li.extend(list(range(len(cur_collect))))
  data[key] = np.array(new_li)
  data[key + '_ind'] = np.array(new_ind_li)


_norm = matplotlib.colors.Normalize(vmin=-70, vmax=0)
_map = cm.ScalarMappable(norm=_norm, cmap=cm.hot)
def to_stype(scalar, correct: bool) -> str:
  return "style='background-color:rgba({:.0f}%, {:.0f}%, {:.0f}%, {:.0f}%); {}' title='{}'".format(
    *([n * 100 for n in _map.to_rgba(scalar)] + (['font-weight:bold' if correct else '']) + [scalar]))


def compute_diversity(tokens_li: List[List[int]]):
  return np.mean([len(set(tokens)) / len(tokens) for tokens in tokens_li])


def analysis_compare(datas: List[Dict], output: str, topk=100):
  os.makedirs(output, exist_ok=True)
  data1 = datas[0]
  data2 = datas[1]
  acc = data1['acc']
  conf_gain = data2['conf'] - data1['conf']

  score = np.array(conf_gain) * (np.array(data1['acc']) * 2 - 1)
  ind = np.argsort(-score)[:topk]
  conf_gain = list(zip(data1['raw_prob'], data2['raw_prob'], data1['conf'], data2['conf'], data2['conf'] - data1['conf']))

  imp_ind = np.arange(len(score))[score >= 0.2]
  same_ind = np.arange(len(score))[np.abs(score) <= 0.01]
  print('#improve {}, #same {}'.format(len(imp_ind), len(same_ind)))

  for group, _ind in [('improve', imp_ind), ('same', same_ind)]:
    print(group)
    for metric in ['target_len', 'input_len', 'target_tokens']:
      if metric == 'target_tokens':
        print(metric, compute_diversity(data1[metric][_ind]), compute_diversity(data2[metric][_ind]))
      else:
        print(metric, np.mean(data1[metric][_ind]), np.mean(data2[metric][_ind]))

  agg_ana_data(data2, 'logprobs', conf_gain)
  display(output, 'improve', data2, ind, conf_gain, acc)


def analysis(datas: List[Dict], output: str, topk=100):
  if len(datas) > 1:
    return analysis_compare(datas, output, topk=topk)
  os.makedirs(output, exist_ok=True)
  data = datas[0]
  conf = data['conf']
  acc = data['acc']
  gap = conf - acc
  plt.hist(gap, bins=20)
  plt.savefig(os.path.join(output, 'gap.png'))
  plt.close()

  under = np.argsort(gap)[:topk]
  over = np.argsort(-gap)[:topk]
  close = np.argsort(np.abs(gap))[:topk]

  agg_ana_data(data, 'logprobs', conf)
  for ind, ind_name in [(under, 'under'), (over, 'over'), (close, 'close')]:
    display(output, ind_name, data, ind, conf, acc)


def display(output: str, prefix: str, data: Dict, ind: List[int], conf: List[str], acc: List[int]):
  def format_conf(conf):
    if type(conf) not in {list, tuple}:
      conf = [conf]
    return ' '.join(['{:.5f}'.format(c) for c in conf])

  for metric, xmin, xmax in [('input_len', 0, 512), ('target_len', 0, 128), ('input_tokens', 1000, 30000), ('target_tokens', 1000, 30000), ('logprobs', 0, 0)]:
    x = data[metric][ind]
    if metric == 'logprobs':
      with open(os.path.join(output, '{}-{}.html'.format(metric, prefix)), 'w') as fout:
        for logprobs, logprobs_ind in zip(x, data['logprobs_ind'][ind]):
          task, inp, _, _, _, _ = logprobs[0]
          fout.write('<div><div>{}</div><div>{}</div>{}</div><hr/>\n'.format(
            '&#x25cf; ' + task,
            inp.replace('\n', '</br>'),
            ''.join([('<div> ' + ('&#9830;' if j == logprobs_ind else '') +  ' &#8594; {}' + '&nbsp;' * 10 + '{}</div>').format(
              format_conf(_conf), ' '.join(['<span {}>{}</span>'.format(to_stype(l, wei == 1), t) for t, l in zip(tgt, lps)])) for j, (_, _, tgt, lps, wei, _conf), in enumerate(logprobs)])))
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
    plt.savefig(os.path.join(output, '{}-{}.png'.format(metric, prefix)))
    plt.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='calibration computation')
  parser.add_argument('--method', type=str, help='metric method', default='macro', choices=['micro', 'macro'])
  parser.add_argument('--mix', type=str, help='mixture', default='uq_sub_test_mix', nargs='+')
  parser.add_argument('--topk', type=int, help='topk options for each question', default=None)
  parser.add_argument('--mn', type=str, help='get m results per n', default=None)
  parser.add_argument('--split', type=str, help='split', default='dev')
  parser.add_argument('--score', type=str, help='score file', nargs='+')
  parser.add_argument('--inp_perp', type=str, help='feature of input perplexity', default=None)
  parser.add_argument('--num_bt', type=int, help='number of translations per example', default=1)
  parser.add_argument('--temp', type=float, help='temperature of softmax', default=1.0)
  parser.add_argument('--xgb', type=str, help='xgb model path', default=None)
  parser.add_argument('--norm', type=str, help='normalization method', default='softmax', choices=['softmax', 'no', 'margin', 'margin_order', 'margin_order_nosm'])
  parser.add_argument('--ana', type=str, help='ana path', default=None)
  args = parser.parse_args()
  if args.mn is not None:
    args.mn = tuple(map(int, args.mn.split(':')))

  # build tasks and mixtures
  build(neg_method='weight', ret_ind=0, ret_method='q-vis')

  if args.ana:
    ana_datas = []
    for i, (mix, score_file) in enumerate(zip(args.mix, args.score)):
      if i == 0 and len(args.mix) > 1:  # the first one use default
        ana_data = acc(mix, [score_file], args.split, ana=args.ana, method=args.method, topk=args.topk, m_per_n=args.mn)
      else:
        ana_data = acc(mix, [score_file], args.split, args.num_bt, args.temp, norm=args.norm, xgb_model_path=args.xgb,
            ana=args.ana, method=args.method, inp_perp=args.inp_perp, topk=args.topk, m_per_n=args.mn)
      ana_datas.append(ana_data)
    print('analysis to {}'.format(args.ana))
    analysis(ana_datas, args.ana, 500)
  else:
    acc(args.mix[0], args.score, args.split, args.num_bt, args.temp, norm=args.norm, xgb_model_path=args.xgb,
        ana=args.ana, method=args.method, inp_perp=args.inp_perp, topk=args.topk, m_per_n=args.mn, fast=True)
