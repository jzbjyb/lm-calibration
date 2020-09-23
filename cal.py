import argparse
from operator import itemgetter
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import t5
import gin
from mesh_tensorflow.transformer import utils
from dataset import build_uq


def acc(mixture: str, score_file: str, split: str='dev'):
  acc_li = []
  conf_li = []

  mix = t5.data.MixtureRegistry.get(mixture)
  ds = mix.get_dataset(split=split, sequence_length={'inputs': 512, 'targets': 512}, shuffle=False, use_filter=False)

  with open(score_file, 'r') as sfin:
    scores = []
    weights = []
    prev_ind = None
    for i, ex in enumerate(ds):
      l = sfin.readline()
      ind = ex['inputs_plaintext'].numpy().decode()
      if prev_ind is not None and ind != prev_ind:
        choice = np.argmax(scores)
        gold = np.argmax(weights)
        acc = choice == gold
        conf = softmax(scores)[choice]
        acc_li.append(acc)
        conf_li.append(conf)
        scores = []
        weights = []
      scores.append(float(l.strip()))
      weights.append(float(ex['weights'].numpy()))
      prev_ind = ind

  print('acc', np.mean(acc_li))

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
  plt.title('acc {:.3f}, ece {:.3f}'.format(np.mean(acc_li), ece))
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
  parser.add_argument('--score', type=str, help='score file')
  args = parser.parse_args()

  # build tasks and mixtures
  build_uq(neg_method='weight')

  acc(args.mix, args.score)
