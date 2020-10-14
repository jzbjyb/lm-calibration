import argparse
from operator import itemgetter
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import t5
from dataset import build


def acc(mixture: str, score_file: str, split: str='dev', num_bt: int=1,
        temp: float=1.0, score_file2: str=None, norm: str = 'softmax'):
  real_acc_li = []
  acc_li = []
  conf_li = []

  mix = t5.data.MixtureRegistry.get(mixture)
  ds = mix.get_dataset_in_order(
    split=split, sequence_length={'inputs': 512, 'targets': 512}, shuffle=False)

  with open(score_file, 'r') as sfin:
    if score_file2 is not None:
      sfin2 = open(score_file2, 'r')
    else:
      sfin2 = None
    scores = []
    weights = []
    prev_ind = None
    for i, ex in enumerate(ds):
      s1 = float(sfin.readline().strip().split('\t', 1)[0])
      s2 = float(sfin2.readline().strip().split('\t', 1)[0]) if sfin2 else 0
      ind = ex['inputs_plaintext'].numpy().decode().split('(a)', 1)[0]
      if prev_ind is not None and ind != prev_ind:
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
          scores = softmax(margin[argind])
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
        choice = np.argmax(_raw_scores)
        real_acc_li.append(int(_weights[choice][0] == 1))
        scores = []
        weights = []
      scores.append(s1 - s2)
      weights.append(float(ex['weights'].numpy()))
      prev_ind = ind

  real_acc = np.mean(real_acc_li)
  print('acc', real_acc)

  num_bins = 20
  margin = 1 / num_bins
  xind = [margin * (i + 0.5) for i in range(num_bins)]

  bins = [[] for _ in range(num_bins)]
  for acc, conf in zip(acc_li, conf_li):
    assert conf >= 0 and conf <= 1, 'confidence out of range'
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

  plt.bar(xind, [np.mean(list(map(itemgetter(1), bin))) for bin in bins], margin)
  plt.title('acc {:.3f}, ece {:.3f}'.format(real_acc, ece))
  plt.ylabel('accuracy')
  plt.xlabel('confidence')
  plt.ylim(0.0, 1.0)
  plt.xlim(0.0, 1.0)
  plt.plot([0, 1], color='red')
  plt.savefig('test.png')
  plt.close()

  print('ece', ece)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='calibration computation')
  parser.add_argument('--mix', type=str, help='mixture', default='uq_sub_test_mix')
  parser.add_argument('--split', type=str, help='split', default='dev')
  parser.add_argument('--score', type=str, help='score file')
  parser.add_argument('--score2', type=str, help='score file', default=None)
  parser.add_argument('--num_bt', type=int, help='number of translations per example', default=1)
  parser.add_argument('--temp', type=float, help='temperature of softmax', default=1.0)
  parser.add_argument('--norm', type=str, help='normalization method', default='softmax', choices=['softmax', 'no', 'margin'])
  args = parser.parse_args()

  # build tasks and mixtures
  build(neg_method='weight')

  acc(args.mix, args.score, args.split, args.num_bt, args.temp, score_file2=args.score2, norm=args.norm)
