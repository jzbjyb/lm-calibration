from typing import List, Tuple, Dict, Set
import os
import csv
from operator import itemgetter
from collections import defaultdict
import random
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
import matplotlib.pyplot as plt
import tensorflow as tf
from dataset.utils import IND2CHAR, CHAR2IND
from dataset.unifiedqa import UNIFIEDQA_GS, UNIFIEDQA_PREP_GS, UNIFIEDQA_PREP_GS_OL, \
  UNIFIEDQA_RAW_GS, UNIFIEDQA_RAW_DECODE_GS, UNIFIEDQA_RAW_DECODE_GS_OL, UNIFIEDQA_PREP_GS_BT, UNIFIEDQA_PREP_GS_BT_REP,\
  DOMAINS, SUB_TEST_DOMAINS, EXT_DOMAINS, MULTI_CHOICE, UNIFIEDQA_PREP_GS_RET_DRQA, UNIFIEDQA_PREP_GS_RET_DRQA_3S
from dataset.unifiedqa import one2multi as one2multi_uq, multi2one
from dataset.test import one2multi as one2multi_test

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


def convert_uq(from_bk, to_bk, domains: List[Tuple[str, List[str]]], format: str='tsv', **kwargs):
  for domain, splits in domains:
    for split in splits:
      in_fname = os.path.join(from_bk, domain, split + '.' + format)
      out_fname = os.path.join(to_bk, domain, split + '.' + format)
      print('{} -> {}'.format(in_fname, out_fname))
      one2multi_uq(in_fname, out_fname, **kwargs)


def multi2one_all(from_bk, to_bk, domains: List[Tuple[str, List[str]]], format: str='tsv', **kwargs):
  for domain, splits in domains:
    for split in splits:
      in_fname = os.path.join(from_bk, domain, split + '.' + format)
      out_fname = os.path.join(to_bk, domain, split + '.' + format)
      print('{} -> {}'.format(in_fname, out_fname))
      multi2one(in_fname, out_fname, **kwargs)


def convert_decoding(from_dir: str, to_dir: str, domains: List[Tuple],
                     split: str, decode_files: List[str], format: str='tsv', beam_size: int=5, use_lower: bool=False):
  count = 0
  defins = [open(df) for df in decode_files]
  try:
    for domain, _ in domains:
      from_file = os.path.join(from_dir, domain, split + '.' + format)
      to_file = os.path.join(to_dir, domain, split + '.' + format)
      print('{} -> {}'.format(from_file, to_file))
      with tf.io.gfile.GFile(from_file, 'r') as fin, tf.io.gfile.GFile(to_file, 'w') as fout:
        for lid, line in enumerate(fin):
          count += 1
          question, answer = line.strip().split('\t')
          question = question.strip()
          answer = answer.strip()
          if use_lower:
            answer = answer.lower()
          decodes: Dict[str, int] = defaultdict(lambda: 0)
          for defin in defins:
            for b in range(beam_size):
              de = defin.readline().strip()
              if use_lower:
                de = de.lower()
              if de == answer or de + '.' == answer or de == answer + '.':  # consider the period
                continue
              decodes[de] += 1
          decodes: List[str] = list(map(itemgetter(0), sorted(decodes.items(), key=lambda x: -x[1])))
          if len(decodes) == 0:
            decodes = [answer] * beam_size
            for did, de in enumerate(decodes):
              fout.write('{}\t{}\t{}\t{}\n'.format(lid, question, de, 'True'))
          else:
            decodes = ([answer] + decodes * beam_size)[:beam_size]  # the correct answer is always the first one
            assert len(decodes) == beam_size, '#decodes {} {} less than {}'.format(len(decodes), decodes, beam_size)
            for did, de in enumerate(decodes):
              fout.write('{}\t{}\t{}\t{}\n'.format(lid, question, de, 'True' if did == 0 else 'False'))
  finally:
    for defin in defins:
      if defin:
        defin.close()
  print('total count {}'.format(count))


def replace_in_ques_bt(from_bk, to_bk, domains: List[Tuple[str, List[str]]], format: str='tsv', split: str='dev'):
  for domain, _ in domains:
    from_file = os.path.join(from_bk, domain, split + '.' + format)
    to_file = os.path.join(to_bk, domain, split + '.' + format)
    print('{} -> {}'.format(from_file, to_file))
    prev_id = None
    prev_answer = None
    with tf.io.gfile.GFile(from_file, 'r') as fin, tf.io.gfile.GFile(to_file, 'w') as fout:
      for lid, line in enumerate(fin):
        id, question, answer, correct = line.strip().split('\t')
        ind, choice = int(id.split('-')[0]), int(id.split('-')[1])
        if id != prev_id:
          prev_answer = answer
          prev_id = id
        choice = IND2CHAR[choice]
        start = question.find('({})'.format(choice))
        start = question.find(prev_answer, start)
        assert start >= 0
        question = question[:start] + answer + question[start + len(prev_answer):]
        fout.write('{}\t{}\t{}\t{}\n'.format(id, question, answer, correct))


def retrieve_aug(from_bk, to_bk, domains: List[Tuple[str, List[str]]], format: str='tsv', splits_restrict: Set[str]={'dev'}, topk=5):
  def get_sent(page, avoid='\t'):
    if page is None:
      return ''
    return page['text'][1].replace(avoid, ' ').replace('\n', ' ').strip() if len(page['text']) > 1 else ''
  from kilt.retrievers import DrQA_tfidf, DPR_connector, BLINK_connector
  from kilt.knowledge_source import KnowledgeSource
  print('load model')
  ranker = DrQA_tfidf.DrQA.from_default_config('drqa')
  print('load mongo')
  ks = KnowledgeSource()
  for domain, splits in domains:
    for split in splits:
      if splits_restrict and split not in splits_restrict:
        continue
      in_fname = os.path.join(from_bk, domain, split + '.' + format)
      out_fname = os.path.join(to_bk, domain, split + '.' + format)
      print('{} -> {}'.format(in_fname, out_fname))
      with tf.io.gfile.GFile(in_fname, 'r') as fin, tf.io.gfile.GFile(out_fname, 'w') as fout:
        lid2data: List[Tuple] = []
        query_data: List[Dict] = []
        for i, line in enumerate(fin):
          lid, ques, ans, correct = line.strip().split('\t')
          lid2data.append((lid, ques, ans, correct))
          query_data.append({'query': ques, 'id': '{}.{}'.format(i, 'q')})
          query_data.append({'query': ans, 'id': '{}.{}'.format(i, 'a')})
        ranker.fed_data(query_data, topk)
        all_doc_id, all_doc_scores, all_query_id, provenance = ranker.run()
        for i, (lid, ques, ans, correct) in tqdm(enumerate(lid2data)):
          qrs = provenance['{}.{}'.format(i, 'q')]
          ars = provenance['{}.{}'.format(i, 'a')]
          qrs = [get_sent(ks.get_page_by_id(int(r['wikipedia_id']))) for r in qrs]
          ars = [get_sent(ks.get_page_by_id(int(r['wikipedia_id']))) for r in ars]
          if len(qrs) < topk:
            qrs += [''] * (topk - len(qrs))
          if len(ars) < topk:
            ars += [''] * (topk - len(ars))
          assert len(qrs) == len(ars) and len(qrs) == topk
          fout.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(lid, ques, ans, correct, '\t'.join(qrs), '\t'.join(ars)))


def truncate_ret_sent(from_bk, to_bk, domains: List[Tuple[str, List[str]]], format: str='tsv', num_sent: int=3):
  empty = 0
  for domain, splits in domains:
    for split in splits:
      in_fname = os.path.join(from_bk, domain, split + '.' + format)
      out_fname = os.path.join(to_bk, domain, split + '.' + format)
      with tf.io.gfile.GFile(in_fname, 'r') as fin, tf.io.gfile.GFile(out_fname, 'w') as fout:
        for i, line in enumerate(fin):
          ls = line.rstrip('\n').split('\t')
          assert len(ls) == 4 + 5 * 2, '{}'.format(line)
          prev, after = ls[:4], ls[4:]
          if len(after[5]) == 0 or len(after[0]) == 0:
            empty += 1
          after = ['. '.join(s.split('. ')[:num_sent]) for s in after]
          fout.write('\t'.join(prev + after) + '\n')
  print('#empty {}'.format(empty))


if __name__ == '__main__':
  # combine('test', 'test.prep')
  # get_input('test.prep', 'test.prep.input')
  # get_input_unifiedqa('test.prep', 'test.prep.unifiedqa_input')

  #acc('test.prep/test.csv', 'test.prep.unifiedqa_input/test_target.txt', 'output/answer/unifiedqa_test_score.txt')

  #convert_uq(UNIFIEDQA_GS, UNIFIEDQA_PREP_GS, DOMAINS)

  #convert_uq(UNIFIEDQA_GS, UNIFIEDQA_PREP_GS_OL, DOMAINS, oneline=True, num_sep=len(MULTI_CHOICE))

  #one2multi_test()

  #convert_decoding(UNIFIEDQA_RAW_GS, UNIFIEDQA_RAW_DECODE_GS + '_uq', EXT_DOMAINS, split='dev',
  #                 decode_file='output/decode/unifiedqa/ext/uq.txt-1100500')

  #convert_decoding(UNIFIEDQA_RAW_GS, UNIFIEDQA_RAW_DECODE_GS + '_uq_ft_softmax', EXT_DOMAINS, split='dev',
  #                 decode_file='output/decode/unifiedqa/ext/uq_ft_softmax.txt-1110000')

  #convert_decoding(UNIFIEDQA_RAW_GS, UNIFIEDQA_RAW_DECODE_GS + '_uq_ft_margin', EXT_DOMAINS, split='dev',
  #                 decode_file='output/decode/unifiedqa/ext/uq_ft_margin.txt-1110000')

  #convert_decoding(UNIFIEDQA_RAW_GS, UNIFIEDQA_RAW_DECODE_GS, EXT_DOMAINS, split='dev', use_lower=True,
  #                 decode_files=['output/decode/unifiedqa/ext/uq.txt-1100500',
  #                               'output/decode/unifiedqa/ext/uq_ft_softmax.txt-1110000',
  #                               'output/decode/unifiedqa/ext/uq_ft_margin.txt-1110000'])

  #replace_in_ques_bt(UNIFIEDQA_PREP_GS_BT, UNIFIEDQA_PREP_GS_BT_REP, SUB_TEST_DOMAINS)

  #multi2one_all(UNIFIEDQA_RAW_DECODE_GS, UNIFIEDQA_RAW_DECODE_GS_OL, EXT_DOMAINS, num_sep=5)

  #retrieve_aug(UNIFIEDQA_PREP_GS, UNIFIEDQA_PREP_GS_RET_DRQA,
  #             SUB_TEST_DOMAINS, splits_restrict={'dev'})
  retrieve_aug(UNIFIEDQA_PREP_GS, UNIFIEDQA_PREP_GS_RET_DRQA,
               SUB_TEST_DOMAINS, splits_restrict={'train', 'test'})
  retrieve_aug(UNIFIEDQA_PREP_GS, UNIFIEDQA_PREP_GS_RET_DRQA,
               list(set(DOMAINS) - set(SUB_TEST_DOMAINS)), splits_restrict={'train', 'dev', 'test'})

  #truncate_ret_sent(UNIFIEDQA_PREP_GS_RET_DRQA, UNIFIEDQA_PREP_GS_RET_DRQA_3S, domains=DOMAINS, num_sent=3)
