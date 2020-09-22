from typing import List
import os
import csv
from operator import itemgetter
from collections import defaultdict
import random
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
from dataset.utils import IND2CHAR, CHAR2IND
from dataset.unifiedqa import UNIFIEDQA_GS, UNIFIEDQA_PREP_GS, DOMAINS, one2multi

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)


def combine(inp_root, out_root):
  os.makedirs(out_root, exist_ok=True)
  for split in ['dev', 'val', 'test']:
    with open(os.path.join(out_root, split + '.csv'), 'w') as fout:
      fout_writer = csv.writer(fout)
      for root, dirs, files in os.walk(os.path.join(inp_root, split)):
        for file in files:
          in_file = os.path.join(root, file)
          with open(in_file) as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
              row = [file] + row
              fout_writer.writerow(row)


def get_input(inp_root, out_root):
  os.makedirs(out_root, exist_ok=True)
  for split in ['dev', 'val', 'test']:
    with open(os.path.join(inp_root, split + '.csv'), 'r') as fin, \
      open(os.path.join(out_root, split + '.txt'), 'w') as fout, \
      open(os.path.join(out_root, split + '_input.txt'), 'w') as pi_fout, \
      open(os.path.join(out_root, split + '_target.txt'), 'w') as po_fout:
      csv_reader = csv.reader(fin)
      for row in csv_reader:
        topic = ' '.join(row[0].rsplit('.', 1)[0].split('_')[:-1])
        ques = row[1].replace('\n', ' ')
        choices = ' '.join(['({}) {}'.format(IND2CHAR[i], c) for i, c in enumerate(row[2:6])]).replace('\n', ' ')
        question = ' '.join(['The following are multiple choice questions about {}.'.format(topic),
                             ques, choices, 'Answer:'])
        fout.write(question + '\n')
        po_fout.write(row[6] + '\n')
        for i, _ in enumerate(row[2:6]):
          pi_fout.write(question + '\n')
          if IND2CHAR[i] != row[6]:
            po_fout.write(IND2CHAR[i] + '\n')


def get_input_unifiedqa(inp_root, out_root):
  os.makedirs(out_root, exist_ok=True)
  for split in ['dev', 'val', 'test']:
    with open(os.path.join(inp_root, split + '.csv'), 'r') as fin, \
      open(os.path.join(out_root, split + '.txt'), 'w') as fout, \
      open(os.path.join(out_root, split + '_input.txt'), 'w') as pi_fout, \
      open(os.path.join(out_root, split + '_target.txt'), 'w') as po_fout:
      csv_reader = csv.reader(fin)
      for row in csv_reader:
        ques = row[1].replace('\n', ' ')
        choices = ' '.join(['({}) {}'.format(IND2CHAR[i], c) for i, c in enumerate(row[2:6])]).replace('\n', ' ')
        question = ' '.join([ques, choices])
        fout.write(question + '\n')
        po_fout.write(row[2 + CHAR2IND[row[6]]].replace('\n', ' ') + '\n')
        for i, c in enumerate(row[2:6]):
          pi_fout.write(question + '\n')
          if IND2CHAR[i] != row[6]:
            po_fout.write(c.replace('\n', ' ') + '\n')


def close(x, y, u, l, thres=0.05):
  return np.abs(x - u) <= thres or np.abs(x - l) <= thres or np.abs(y - u) <= thres or np.abs(y - l) <= thres


def acc(csv_file, answer_file, logit_file):
  acc_li = []
  conf_li = []
  topic2li = defaultdict(lambda: {'acc': [], 'conf': []})
  with open(csv_file, 'r') as cfin, open(answer_file, 'r') as afin, open(logit_file, 'r') as lfin:
    csv_reader = csv.reader(cfin)
    scores = []
    for i, l in enumerate(lfin):
      if i % 4 == 0 and len(scores) > 0:
        row = next(csv_reader)
        choice = np.argmax(scores)
        acc = choice == 0
        conf = softmax(scores)[choice]
        acc_li.append(acc)
        conf_li.append(conf)
        topic2li[row[0]]['acc'].append(acc)
        topic2li[row[0]]['conf'].append(conf)
        scores = []
      scores.append(float(l.strip()))

  print('acc', np.mean(acc_li))

  num_bins = 10
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

  topic2li = list(topic2li.items())
  random.shuffle(topic2li)
  topics = [(np.mean(c['conf']), np.mean(c['acc']), k) for k, c in topic2li]
  x, y, labels = list(zip(*topics))
  upper = max(np.max(x), np.max(y))
  lower = min(np.min(x), np.min(y))
  fig, ax = plt.subplots()
  plt.ylabel('accuracy')
  plt.xlabel('confidence')
  plt.ylim(lower, upper)
  plt.xlim(lower, upper)
  ax.scatter(x, y)
  count = 0
  for i, label in enumerate(labels):
    if close(x[i], y[i], upper, lower) and count <= 5:
      count += 1
      ax.annotate(' '.join(label.rsplit('.', 1)[0].split('_')[:-1]), (x[i] - 0.1, y[i]))
  plt.savefig('test2.png')
  plt.close()


def convert_uq(domain: str, splits: List[str], format: str='tsv'):
  for split in splits:
    in_fname = os.path.join(UNIFIEDQA_GS, domain, split + '.' + format)
    out_fname = os.path.join(UNIFIEDQA_PREP_GS, domain, split + '.' + format)
    print('{} -> {}'.format(in_fname, out_fname))
    one2multi(in_fname, out_fname)


if __name__ == '__main__':
  # combine('test', 'test.prep')
  # get_input('test.prep', 'test.prep.input')
  # get_input_unifiedqa('test.prep', 'test.prep.unifiedqa_input')

  #acc('test.prep/test.csv', 'test.prep.unifiedqa_input/test_target.txt', 'output/answer/unifiedqa_test_score.txt')

  for domain, splits in DOMAINS:
    convert_uq(domain, splits)
