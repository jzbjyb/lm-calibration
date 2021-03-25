from typing import List, Tuple, Dict, Set
import os
import sys
import csv
from operator import itemgetter
from collections import defaultdict
import random
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
import matplotlib.pyplot as plt
import tensorflow as tf
from dataset.utils import IND2CHAR, CHAR2IND, normalize_answer
from dataset.unifiedqa import UNIFIEDQA_GS, UNIFIEDQA_PREP_GS, UNIFIEDQA_PREP_GS_OL, \
  UNIFIEDQA_RAW_GS, UNIFIEDQA_RAW_DECODE_GS, UNIFIEDQA_RAW_DECODE_GS_OL, UNIFIEDQA_PREP_GS_BT, UNIFIEDQA_PREP_GS_BT_REP,\
  DOMAINS, SUB_TEST_DOMAINS, TEST_DOMAINS, EXT_DOMAINS, MULTI_CHOICE, UNIFIEDQA_PREP_GS_RET_DRQA, UNIFIEDQA_PREP_GS_RET_DRQA_3S, \
  UNIFIEDQA_RAW_DECODE_GS_OL_ANS, UNIFIEDQA_RAW_DECODE_GS_ANS, UNIFIEDQA_RAW_DECODE_GS_ANS_NO, \
  UNIFIEDQA_RAW_DECODE_GS_OL_ANS_NO, UNIFIEDQA_PREP_GS_RET_DRQA_3S_BT_REP, \
  UNIFIEDQA_PREP_GS_BT_DEDUP, UNIFIEDQA_PREP_GS_BT_DEDUP_REP, UNIFIEDQA_PREP_GS_RET_DRQA_3S_BT_DEDUP_REP
from dataset.unifiedqa import UNIFIEDQA_RAW_DECODE_UQ3B_GS, UNIFIEDQA_RAW_DECODE_UQ3B_GS_OL, \
  UNIFIEDQA_RAW_DECODE_UQ3B_GS_RET_DRQA, UNIFIEDQA_RAW_DECODE_UQ3B_GS_RET_DRQA_3S, \
  UNIFIEDQA_RAW_DECODE_UQ3B_GS_BT, UNIFIEDQA_RAW_DECODE_UQ3B_GS_RET_DRQA_3S_BT, \
  UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS, UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_OL, UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_RET_DRQA, \
  UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_RET_DRQA_3S, UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_BT, \
  UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_RET_DRQA_3S_BT, UNIFIEDQA_RAW_DUP_GS, \
  UNIFIEDQA_PREP_GS_BT_DEDUP_TOP10, UNIFIEDQA_PREP_GS_BT_DEDUP_TOP10_REP, \
  UNIFIEDQA_PREP_GS_BT_DEDUP_TOP20, UNIFIEDQA_PREP_GS_BT_DEDUP_TOP20_REP
from dataset.unifiedqa import UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS, UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_OL, \
  UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_RET_DRQA, UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_RET_DRQA_3S, \
  UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_BT, UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_RET_DRQA_3S_BT, \
  UNIFIEDQA_RAW_FIRST_DECODE_UQ3B_GS, UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_GS, \
  UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS, UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS_RET_DRQA, \
  UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS_RET_DRQA_3S, UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_GS_OL, \
  UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS_RET_DRQA_3S_BT, UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS_BT
from dataset.unifiedqa import MH_DOMAINS
from dataset.unifiedqa import one2multi as one2multi_uq, multi2one
from dataset.test import one2multi as one2multi_test, TEST_PREP_GS, TEST_PREP_GS_RET_DRQA, TEST_PREP_GS_RET_DRQA_3S, \
  TEST_PREP_GS_BT, TEST_PREP_GS_BT_REP, TEST_PREP_GS_RET_DRQA_3S_BT_REP, TEST_PREP_GS_BT_DEDUP, \
  TEST_PREP_GS_BT_DEDUP_REP, TEST_PREP_GS_RET_DRQA_3S_BT_DEDUP_REP
from dataset.test import DOMAINS as MT_TEST_DOMAINS
from t5.data.utils import get_default_vocabulary

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


def dedup_list(texts: List[str], min_count: int):
  nodup: List[str] = []
  remain: List[str] = []
  for text in texts:
    no = True
    if len(nodup) > 0 and len(text) > len(nodup[0]) * 5:
      no = False
    else:
      for s in nodup:
        _text = text.rstrip('.')
        _s = s.rstrip('.')
        if _text in _s or _s in _text:
          no = False
          break
    if no:
      nodup.append(text)
    else:
      remain.append(text)
  return nodup + remain, len(nodup) >= min_count


def convert_decoding(from_dir: str, to_dir: str, domains: List[Tuple],
                     split: str, decode_files: List[str], format: str='tsv', beam_size: int=5, keep_size: int=5,
                     use_lower: bool=False, add_ans: bool=False, dedup: bool=False):
  count = has_enough_count = 0
  defins = [open(df) for df in decode_files]
  try:
    for domain, _ in domains:
      from_file = os.path.join(from_dir, domain, split + '.' + format)
      to_file = os.path.join(to_dir, domain, split + '.' + format)
      print('{} -> {}'.format(from_file, to_file))
      with tf.io.gfile.GFile(from_file, 'r') as fin, tf.io.gfile.GFile(to_file, 'w') as fout:
        for lid, line in enumerate(fin):
          count += 1
          question, answer = line.rstrip('\n').split('\t')
          question = question.strip()
          answer = answer.strip()
          if use_lower:
            answer = answer.lower()
          decodes: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
          real_decodes: Dict[str, int] = defaultdict(lambda: 0)
          for defin in defins:
            for b in range(beam_size):
              de = defin.readline().strip()
              if use_lower:
                de = de.lower()
              real_decodes[de] += 1
              if de == answer or de + '.' == answer or de == answer + '.':  # consider the period
                continue
              if de not in decodes:
                decodes[de][1] = b
              decodes[de][0] += 1
          decodes: List[str] = list(map(itemgetter(0), sorted(decodes.items(), key=lambda x: (-x[1][0], x[1][1]))))
          real_decodes: List[str] = list(map(itemgetter(0), sorted(real_decodes.items(), key=lambda x: -x[1])))[:keep_size]
          if add_ans:
            question = '{} \\n {}'.format(question, ' '.join(['({}) {}'.format(IND2CHAR[i], a) for i, a in enumerate(real_decodes)]))
          if len(decodes) == 0:
            decodes = [answer] * keep_size
            for did, de in enumerate(decodes):
              fout.write('{}\t{}\t{}\t{}\n'.format(lid, question, de, 'True'))
          else:
            if dedup:
              decodes, has_enough = dedup_list([answer] + decodes * keep_size, min_count=keep_size)
              has_enough_count += int(has_enough)
              decodes = decodes[:keep_size]
            else:
              decodes = ([answer] + decodes * keep_size)[:keep_size]  # the correct answer is always the first one
            assert len(decodes) == keep_size, '#decodes {} {} less than {}'.format(len(decodes), decodes, keep_size)
            for did, de in enumerate(decodes):
              fout.write('{}\t{}\t{}\t{}\n'.format(lid, question, de, 'True' if did == 0 else 'False'))
  finally:
    for defin in defins:
      if defin:
        defin.close()
  print('total count {}, {} has enough dedup'.format(count, has_enough_count))


def convert_first_token_decoding(from_dir: str, to_dir: str, domains: List[Tuple], split: str, score_file: str, format: str='tsv', num_spans: int=10, span_len: int=128):
  vocab = get_default_vocabulary()
  all_count = miss_count = multi_first_count = 0
  with open(score_file, 'r') as score_fin:
    for domain, _ in domains:
      from_file = os.path.join(from_dir, domain, split + '.' + format)
      to_file = os.path.join(to_dir, domain, split + '.' + format)
      print('{} -> {}'.format(from_file, to_file))
      with tf.io.gfile.GFile(from_file, 'r') as fin, tf.io.gfile.GFile(to_file, 'w') as fout:
        for lid, line in enumerate(fin):
          question, answer = line.rstrip('\n').split('\t')
          question = question.strip()
          answer = answer.strip()
          _, inp_tokens, tgt_tokens, _, first_ind, first_prob = score_fin.readline().rstrip('\n').split('\t')[:6]
          inp_tokens = [int(i) for i in inp_tokens.split(',0', 1)[0].split(',')]
          inptok2pos = defaultdict(list)
          for i, t in enumerate(inp_tokens):
            inptok2pos[t].append(i)
          first_ind = [int(i) for i in first_ind.split(',')]
          first_prob = [float(i) for i in first_prob.split(',')]
          cand_set = set()
          cand_li = []
          prob_li = []
          for i, (ind, prob) in enumerate(zip(first_ind, first_prob)):
            if ind not in inptok2pos:
              continue
            for pos in inptok2pos[ind]:
              cand = vocab.decode(inp_tokens[pos:pos + span_len])
              if cand not in cand_set:
                cand_set.add(cand)
                cand_li.append(cand)
                prob_li.append(prob)
              if len(cand_set) >= num_spans + 1:
                break
            if len(cand_set) >= num_spans + 1:
              break
          assert len(cand_set) == num_spans + 1
          miss_count + int(prob_li[-1] >= 0.1)
          multi_first_count += int(i >= 1)
          all_count += 1
          for cand in cand_li[:-1]:
            fout.write('{}\t{}\t{}\t{}\n'.format(lid, question, cand, 'False'))
  print('all {} miss {} multi first token {}'.format(all_count, miss_count, multi_first_count))


def get_topk_decodes(answer: str, candidate_li: List[List[int]], logprob_li: List[List[float]], end_logprob_li: List[List[float]], vocab, num_spans: int, span_len: int, remove_dup: bool=True, end_sym: int=1, use_gold: bool=True):
  can2prob: Dict[str, float] = defaultdict(lambda: 1000)
  for candidate, logprob, end_logprob in zip(candidate_li, logprob_li, end_logprob_li):
    prev_lp = 0
    for i in range(1, min(len(candidate) - 1, span_len)):
      can = vocab.decode(candidate[:i] + [end_sym])
      lp = prev_lp + logprob[i - 1] + end_logprob[i]
      prev_lp += logprob[i - 1]
      can2prob[can] = min(can2prob[can], lp)
  answer_set = {answer.lower(), answer.lower().rstrip('.'), answer.lower() + '.'}
  if use_gold:
    # remove answer
    for a in answer_set:
      if a in can2prob:
        del can2prob[a]
  # rank
  can2prob = sorted(can2prob.items(), key=lambda x: -x[1])
  if remove_dup:
    # remove dup
    dedup_can2prob = []
    dup_can2prob = []
    canset: Set[str] = {answer.lower()} if use_gold else set()
    for c, p in can2prob:
      dup = False
      for _c in canset:
        if _c in c or c in _c:
          dup = True
          break
      if not dup:
        canset.add(c)
        dedup_can2prob.append((c, p))
        if len(dedup_can2prob) >= num_spans:
          break
      else:
        dup_can2prob.append((c, p))
    can2prob = dedup_can2prob + dup_can2prob
  if use_gold:
    can2prob = can2prob[:num_spans - 1]
    return [(answer, 1)] + [(c, 0) for c, p in can2prob]
  else:
    return [(c, int(c in answer_set)) for c, p in can2prob][:num_spans]


def convert_first_token_decoding_topk(gold_dir: str, from_dir: str, to_dir: str, domains: List[Tuple], split: str, score_file: str, format: str='tsv', num_spans: int=5, span_len: int=20, use_gold: bool=True):
  vocab = get_default_vocabulary()
  with open(score_file, 'r') as score_fin:
    for domain, _ in domains:
      count_correct = count = 0
      gold_file = os.path.join(gold_dir, domain, split + '.' + format)
      from_file = os.path.join(from_dir, domain, split + '.' + format)
      to_file = os.path.join(to_dir, domain, split + '.' + format)
      print('{} -> {}'.format(from_file, to_file))

      prev_lid = None
      candidate_li: List[List[int]] = []
      logprob_li: List[List[float]] = []
      end_logprob_li: List[List[float]] = []
      with tf.io.gfile.GFile(gold_file, 'r') as gold_fin, tf.io.gfile.GFile(from_file, 'r') as fin, tf.io.gfile.GFile(to_file, 'w') as fout:
        for line in tqdm(fin):
          lid, question, candidate, _ = line.rstrip('\n').split('\t')

          if prev_lid is not None and lid != prev_lid:
            question, answer = gold_fin.readline().rstrip('\n').split('\t')
            decodes = get_topk_decodes(answer, candidate_li, logprob_li, end_logprob_li, vocab, num_spans=num_spans, span_len=span_len, use_gold=use_gold)
            assert len(decodes) == num_spans
            for i, (de, correct) in enumerate(decodes):
              fout.write('{}\t{}\t{}\t{}\n'.format(prev_lid, question, de, 'True' if correct else 'False'))
              count += 1
              count_correct += correct
            candidate_li = []
            logprob_li = []
            end_logprob_li = []

          _, _, tgt_tokens, logprob, first_ind, first_prob, end_logprob = score_fin.readline().rstrip('\n').split('\t')[:7]
          tgt_tokens = [int(i) for i in tgt_tokens.split(',0', 1)[0].split(',')]
          logprob = [float(i) for i in logprob.split(',')][:len(tgt_tokens)]
          end_logprob = [float(i) for i in end_logprob.split(',')][:len(tgt_tokens)]

          prev_lid = lid
          candidate_li.append(tgt_tokens)
          logprob_li.append(logprob)
          end_logprob_li.append(end_logprob)

        if len(candidate_li) > 0:
          question, answer = gold_fin.readline().rstrip('\n').split('\t')
          decodes = get_topk_decodes(answer, candidate_li, logprob_li, end_logprob_li, vocab, num_spans=num_spans, span_len=span_len, use_gold=use_gold)
          assert len(decodes) == num_spans
          for i, (de, correct) in enumerate(decodes):
            fout.write('{}\t{}\t{}\t{}\n'.format(lid, question, de, 'True' if correct else 'False'))
            count += 1
            count_correct += correct
      print('#correct {} #all {}'.format(count_correct, count))


def replace_in_ques_bt(from_bk, to_bk, domains: List[Tuple[str, List[str]]], format: str='tsv', splits_restrict: Set[str]=None):
  for domain, splits in domains:
    for split in splits:
      if splits_restrict is not None and split not in splits_restrict:
        continue
      from_file = os.path.join(from_bk, domain, split + '.' + format)
      to_file = os.path.join(to_bk, domain, split + '.' + format)
      print('{} -> {}'.format(from_file, to_file))
      prev_id = None
      prev_answer = None
      ab_count = 0
      with tf.io.gfile.GFile(from_file, 'r') as fin, tf.io.gfile.GFile(to_file, 'w') as fout:
        for lid, line in enumerate(fin):
          id, question, answer, correct = line.strip().split('\t')
          ids = id.split('-')[-2:]
          ind, choice = int(ids[0]), int(ids[1])
          if id != prev_id:
            prev_answer = answer
            prev_id = id
          start = question.find('({})'.format(IND2CHAR[choice]))
          start = question.find(prev_answer, start)
          if start < 0:
            start = question.find('({})'.format(IND2CHAR[choice]))
            end = question.find('({})'.format(IND2CHAR[choice + 1]))
            if end < 0:
              end = len(question)
            assert start >= 0 and end >= 0
            question = question[:start] + '({}) {} '.format(IND2CHAR[choice], answer) + question[end:]
            ab_count += 1
          else:
            question = question[:start] + answer + question[start + len(prev_answer):]
          fout.write('{}\t{}\t{}\t{}\n'.format(id, question, answer, correct))
        print('abnormal count {}'.format(ab_count))


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
      print('{} -> {}'.format(in_fname, out_fname), flush=True)
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
        for i, (lid, ques, ans, correct) in tqdm(enumerate(lid2data), desc='collect retrieved wiki articles'):
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
  def truncate(sent):
    ss = sent.split('. ')
    if len(ss) <= num_sent:
      return sent
    return '. '.join(ss[:num_sent]) + '.'
  for domain, splits in domains:
    for split in splits:
      in_fname = os.path.join(from_bk, domain, split + '.' + format)
      out_fname = os.path.join(to_bk, domain, split + '.' + format)
      print('{} -> {}'.format(in_fname, out_fname))
      with tf.io.gfile.GFile(in_fname, 'r') as fin, tf.io.gfile.GFile(out_fname, 'w') as fout:
        for i, line in enumerate(fin):
          ls = line.rstrip('\n').split('\t')
          assert len(ls) == 4 + 5 * 2, '{}'.format(line)
          prev, after = ls[:4], ls[4:]
          if len(after[5]) == 0 or len(after[0]) == 0:
            empty += 1
          after = [truncate(s) for s in after]
          fout.write('\t'.join(prev + after) + '\n')
  print('#empty {}'.format(empty))


def convert_ol_to_add_answers(from_bk, to_bk, domains: List[Tuple[str, List[str]]], format: str='tsv', multiline: bool=False):
  for domain, splits in domains:
    for split in splits:
      in_fname = os.path.join(from_bk, domain, split + '.' + format)
      out_fname = os.path.join(to_bk, domain, split + '.' + format)
      print(in_fname, out_fname)
      with tf.io.gfile.GFile(in_fname, 'r') as fin, tf.io.gfile.GFile(out_fname, 'w') as fout:
        for i, line in enumerate(fin):
          ls = line.rstrip('\n').split('\t')
          ans = [a for i, a in enumerate(ls[2:]) if i % 2 == 0]
          shuf_ans = np.random.choice(ans, len(ans), replace=False)
          ls[1] = '{} \\n {}'.format(ls[1], ' '.join(['({}) {}'.format(IND2CHAR[i], a) for i, a in enumerate(shuf_ans)]))
          if multiline:
            for i, an in enumerate(ans):
              fout.write('{}\t{}\t{}\t{}\n'.format(ls[0], ls[1], an, 'True' if i == 0 else 'False'))
          else:
            fout.write('{}\n'.format('\t'.join(ls)))


def combine_ret_bt(from_bk_ret, from_bk_bt, to_bk, domains: List[Tuple[str, List[str]]],
                   format: str='tsv', splits_restrict: Set[str]={'dev'}, num_bt: int=5):
  for domain, splits in domains:
    for split in splits:
      if splits_restrict and split not in splits_restrict:
        continue
      in_fname_ret = os.path.join(from_bk_ret, domain, split + '.' + format)
      in_fname_bt = os.path.join(from_bk_bt, domain, split + '.' + format)
      out_fname = os.path.join(to_bk, domain, split + '.' + format)
      print(in_fname_bt, in_fname_ret, out_fname)
      with tf.io.gfile.GFile(in_fname_ret, 'r') as fin_ret, \
        tf.io.gfile.GFile(in_fname_bt, 'r') as fin_bt, \
        tf.io.gfile.GFile(out_fname, 'w') as fout:
        for i, line_ret in enumerate(fin_ret):
          rets = line_ret.rstrip('\n').split('\t')[4:]
          nb = num_bt
          while nb > 0:
            fout.write(fin_bt.readline().rstrip('\n') + '\t' + '\t'.join(rets) + '\n')
            nb -= 1


def fix_test(fom_bk: str, to_bk: str, domains: List[Tuple[str, List]], format: str='tsv', num_bt: int=0):
  for domain, splits in domains:
    for split in splits:
      in_fname = os.path.join(fom_bk, domain, split + '.' + format)
      out_fname = os.path.join(to_bk, domain, split + '.' + format)
      print(in_fname, out_fname)
      with tf.io.gfile.GFile(in_fname, 'r') as fin, tf.io.gfile.GFile(out_fname, 'w') as fout:
        prev_inp = None
        qid = 0
        aid = 0
        for l in fin:
          ls = l.rstrip('\n').split('\t')
          ind = ls[0].split('-')[0]
          inp = ls[1]
          if prev_inp is not None and inp != prev_inp:
            qid += 1
            aid = 0
          ind = '{}-{}'.format(ind, qid)
          if num_bt:
            ind = '{}-{}'.format(ind, aid // num_bt)
          prev_inp = inp
          aid += 1
          fout.write('\t'.join([ind, inp] + ls[2:]) + '\n')


def duplicate(fom_bk: str, to_bk: str, domains: List[Tuple[str, List]], format: str='tsv', dup_count: int=1):
  for domain, splits in domains:
    for split in splits:
      in_fname = os.path.join(fom_bk, domain, split + '.' + format)
      out_fname = os.path.join(to_bk, domain, split + '.' + format)
      print(in_fname, out_fname)
      with tf.io.gfile.GFile(in_fname, 'r') as fin, tf.io.gfile.GFile(out_fname, 'w') as fout:
        for l in fin:
          for i in range(dup_count):
            fout.write(l)


def get_top_bt(fom_bk: str, bt_bk: str, to_bk: str, domains: List[Tuple[str, List]], format: str='tsv', total: int=1, topk: int=1, splits_restrict: Set[str]=None):
  for domain, splits in domains:
    for split in splits:
      if splits_restrict and split not in splits_restrict:
        continue
      in_fname = os.path.join(fom_bk, domain, split + '.' + format)
      bt_fname = os.path.join(bt_bk, domain, split + '.' + format + '.raw')
      out_fname = os.path.join(to_bk, domain, split + '.' + format)
      all_count = out_of_bound_count = 0
      with tf.io.gfile.GFile(in_fname, 'r') as fin, \
        tf.io.gfile.GFile(bt_fname, 'r') as bt_fin, \
        tf.io.gfile.GFile(out_fname, 'w') as fout:
        for l in fin:
          _, question, answer, correct = l.rstrip('\n').split('\t')

          # read bts
          bts: Dict[str, int] = defaultdict(lambda: 0)
          for i in range(total):
            qid, _, bt, _ = bt_fin.readline().rstrip('\n').split('\t')
            bts[bt] += 1

          if answer in bts:
            bts[answer] -= (total - topk)
            if bts[answer] <= 0:
              del bts[answer]
            else:
              out_of_bound_count += 1

          # output
          fout.write('{}\t{}\t{}\t{}\n'.format(qid, question, answer, correct))

          bts = sorted(bts.items(), key=lambda x: -x[1])
          c = topk - 1
          ind = 0
          all_count += 1
          out_of_bound = False
          while c > 0:
            if ind >= len(bts) or bts[ind][1] <= 0:
              if not out_of_bound:
                out_of_bound_count += 1
                out_of_bound = True
              ind = 0
            assert bts[ind][1] > 0, 'back translations are exhausted'
            fout.write('{}\t{}\t{}\t{}\n'.format(qid, question, bts[ind][0], correct))
            bts[ind] = (bts[ind][0], bts[ind][1] - 1)
            ind += 1
            c -= 1
      print(in_fname, out_fname, 'outofbound', out_of_bound_count, all_count)


def reducehop_first(raw_reducehop_dir: str, reducehop_first_dir: str, num_hop: int=2):
  for root, dir, files in os.walk(raw_reducehop_dir):
    for file in files:
      from_file = os.path.join(root, file)
      to_file = from_file.replace(raw_reducehop_dir, reducehop_first_dir)
      os.makedirs(os.path.dirname(to_file), exist_ok=True)
      with open(from_file, 'r') as fin, open(to_file, 'w') as fout:
        for i, l in enumerate(fin):
          if i % num_hop == 0:
            fout.write(l)


def reducehop_second(raw_reducehop_dir: str, prediction_path: str, reducehop_second_dir: str,
                     num_hop: int=2, domains: List[Tuple]=None, split: str='dev'):
  with open(prediction_path, 'r') as pfin:
    for domain, splits in domains:
      from_file = os.path.join(raw_reducehop_dir, domain, split + '.tsv')
      to_file = os.path.join(reducehop_second_dir, domain, split + '.tsv')
      os.makedirs(os.path.dirname(to_file), exist_ok=True)
      with open(from_file, 'r') as ffin, open(to_file, 'w') as fout:
        for i, l in enumerate(ffin):
          if i % num_hop == 1:
            l = l.rstrip('\n').split('\t')
            p = pfin.readline().strip()
            l[1] = p
            fout.write('\t'.join(l) + '\n')


if __name__ == '__main__':
  task = sys.argv[1]

  if task == 'bt':
    replace_in_ques_bt(UNIFIEDQA_PREP_GS_BT, UNIFIEDQA_PREP_GS_BT_REP, DOMAINS, splits_restrict={'dev'})
    replace_in_ques_bt(TEST_PREP_GS_BT, TEST_PREP_GS_BT_REP, MT_TEST_DOMAINS, splits_restrict={'test'})
    replace_in_ques_bt(UNIFIEDQA_PREP_GS_BT_DEDUP, UNIFIEDQA_PREP_GS_BT_DEDUP_REP, DOMAINS, splits_restrict={'dev'})
    replace_in_ques_bt(TEST_PREP_GS_BT_DEDUP, TEST_PREP_GS_BT_DEDUP_REP, MT_TEST_DOMAINS, splits_restrict={'test'})

  if task == 'bt_topk':
    get_top_bt(UNIFIEDQA_PREP_GS, UNIFIEDQA_PREP_GS_BT_DEDUP, UNIFIEDQA_PREP_GS_BT_DEDUP_TOP10, DOMAINS, total=100, topk=10, splits_restrict={'dev'})
    replace_in_ques_bt(UNIFIEDQA_PREP_GS_BT_DEDUP_TOP10, UNIFIEDQA_PREP_GS_BT_DEDUP_TOP10_REP, DOMAINS, splits_restrict={'dev'})
    get_top_bt(UNIFIEDQA_PREP_GS, UNIFIEDQA_PREP_GS_BT_DEDUP, UNIFIEDQA_PREP_GS_BT_DEDUP_TOP20, DOMAINS, total=100, topk=20, splits_restrict={'dev'})
    replace_in_ques_bt(UNIFIEDQA_PREP_GS_BT_DEDUP_TOP20, UNIFIEDQA_PREP_GS_BT_DEDUP_TOP20_REP, DOMAINS, splits_restrict={'dev'})

  if task == 'ret':
    retrieve_aug(UNIFIEDQA_PREP_GS, UNIFIEDQA_PREP_GS_RET_DRQA, DOMAINS, splits_restrict={'train', 'dev', 'test'})
    truncate_ret_sent(UNIFIEDQA_PREP_GS_RET_DRQA, UNIFIEDQA_PREP_GS_RET_DRQA_3S, domains=DOMAINS, num_sent=3)
    retrieve_aug(TEST_PREP_GS, TEST_PREP_GS_RET_DRQA, MT_TEST_DOMAINS, splits_restrict={'val', 'dev', 'test'})
    truncate_ret_sent(TEST_PREP_GS_RET_DRQA, TEST_PREP_GS_RET_DRQA_3S, domains=MT_TEST_DOMAINS, num_sent=3)
    retrieve_aug(UNIFIEDQA_RAW_DECODE_UQ3B_GS, UNIFIEDQA_RAW_DECODE_UQ3B_GS_RET_DRQA, EXT_DOMAINS, splits_restrict={'dev'})
    truncate_ret_sent(UNIFIEDQA_RAW_DECODE_UQ3B_GS_RET_DRQA, UNIFIEDQA_RAW_DECODE_UQ3B_GS_RET_DRQA_3S, domains=EXT_DOMAINS, num_sent=3)
    retrieve_aug(UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS, UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_RET_DRQA, EXT_DOMAINS, splits_restrict={'dev'})
    truncate_ret_sent(UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_RET_DRQA, UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_RET_DRQA_3S, domains=EXT_DOMAINS, num_sent=3)
    retrieve_aug(UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS, UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_RET_DRQA, EXT_DOMAINS, splits_restrict={'dev'})
    truncate_ret_sent(UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_RET_DRQA, UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_RET_DRQA_3S, domains=EXT_DOMAINS, num_sent=3)
    retrieve_aug(UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS, UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS_RET_DRQA, EXT_DOMAINS, splits_restrict={'dev'})
    truncate_ret_sent(UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS_RET_DRQA, UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS_RET_DRQA_3S, domains=EXT_DOMAINS, num_sent=3)

  if task == 'combine_ret_bt':
    combine_ret_bt(UNIFIEDQA_PREP_GS_RET_DRQA_3S, UNIFIEDQA_PREP_GS_BT_REP,
                   UNIFIEDQA_PREP_GS_RET_DRQA_3S_BT_REP, DOMAINS,
                   splits_restrict={'dev'}, num_bt=5)
    combine_ret_bt(UNIFIEDQA_PREP_GS_RET_DRQA_3S, UNIFIEDQA_PREP_GS_BT_DEDUP_REP,
                   UNIFIEDQA_PREP_GS_RET_DRQA_3S_BT_DEDUP_REP, DOMAINS,
                   splits_restrict={'dev'}, num_bt=5)
    combine_ret_bt(TEST_PREP_GS_RET_DRQA_3S, TEST_PREP_GS_BT_REP,
                   TEST_PREP_GS_RET_DRQA_3S_BT_REP, MT_TEST_DOMAINS,
                   splits_restrict={'test'}, num_bt=5)
    combine_ret_bt(TEST_PREP_GS_RET_DRQA_3S, TEST_PREP_GS_BT_DEDUP_REP,
                   TEST_PREP_GS_RET_DRQA_3S_BT_DEDUP_REP, MT_TEST_DOMAINS,
                   splits_restrict={'test'}, num_bt=5)
    combine_ret_bt(UNIFIEDQA_RAW_DECODE_UQ3B_GS_RET_DRQA_3S, UNIFIEDQA_RAW_DECODE_UQ3B_GS_BT,
                   UNIFIEDQA_RAW_DECODE_UQ3B_GS_RET_DRQA_3S_BT, EXT_DOMAINS,
                   splits_restrict={'dev'}, num_bt=5)
    combine_ret_bt(UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_RET_DRQA_3S, UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_BT,
                   UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_RET_DRQA_3S_BT, EXT_DOMAINS,
                   splits_restrict={'dev'}, num_bt=5)
    combine_ret_bt(UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_RET_DRQA_3S, UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_BT,
                   UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_RET_DRQA_3S_BT, EXT_DOMAINS,
                   splits_restrict={'dev'}, num_bt=5)
    combine_ret_bt(UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS_RET_DRQA_3S, UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS_BT,
                   UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS_RET_DRQA_3S_BT, EXT_DOMAINS,
                   splits_restrict={'dev'}, num_bt=5)

  if task == 'fix_test':
    fix_test(TEST_PREP_GS + '.bak', TEST_PREP_GS, MT_TEST_DOMAINS)
    fix_test(TEST_PREP_GS_RET_DRQA + '.bak', TEST_PREP_GS_RET_DRQA, MT_TEST_DOMAINS)
    fix_test(TEST_PREP_GS_BT + '.bak', TEST_PREP_GS_BT, MT_TEST_DOMAINS, num_bt=5)
    fix_test(TEST_PREP_GS_BT_DEDUP + '.bak', TEST_PREP_GS_BT_DEDUP, MT_TEST_DOMAINS, num_bt=5)

  if task == 'decode':
    convert_decoding(UNIFIEDQA_RAW_GS, UNIFIEDQA_RAW_DECODE_UQ3B_GS, EXT_DOMAINS, split='dev', use_lower=True,
                     decode_files=['output/decode/unifiedqa/ext_dev/uq_bs20.txt-1100500'], beam_size=20, keep_size=5)
    convert_decoding(UNIFIEDQA_RAW_GS, UNIFIEDQA_RAW_DECODE_UQ3B_GS, EXT_DOMAINS, split='train', use_lower=True,
                     decode_files=['output/decode/unifiedqa/ext_train/uq_bs20.txt-1100500'], beam_size=20, keep_size=5)
    multi2one_all(UNIFIEDQA_RAW_DECODE_UQ3B_GS, UNIFIEDQA_RAW_DECODE_UQ3B_GS_OL, EXT_DOMAINS, num_sep=5)

  if task == 'decode_dedup':
    convert_decoding(UNIFIEDQA_RAW_GS, UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS, EXT_DOMAINS, split='dev', use_lower=True,
                     decode_files=['output/decode/unifiedqa/ext_dev/uq_bs20.txt-1100500'], beam_size=20, keep_size=5, dedup=True)
    convert_decoding(UNIFIEDQA_RAW_GS, UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS, EXT_DOMAINS, split='train', use_lower=True,
                     decode_files=['output/decode/unifiedqa/ext_train/uq_bs20.txt-1100500'], beam_size=20, keep_size=5, dedup=True)
    multi2one_all(UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS, UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_OL, EXT_DOMAINS, num_sep=5)

  if task == 'decode_sample':
    convert_decoding(UNIFIEDQA_RAW_GS, UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS, EXT_DOMAINS, split='dev', use_lower=True,
                     decode_files=['output/decode/unifiedqa/ext_dev/uq_dup.txt-1100500'], beam_size=10, keep_size=5, dedup=True)
    convert_decoding(UNIFIEDQA_RAW_GS, UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS, EXT_DOMAINS, split='train', use_lower=True,
                     decode_files=['output/decode/unifiedqa/ext_train/uq_dup.txt-1100500'], beam_size=10, keep_size=5, dedup=True)
    multi2one_all(UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS, UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_OL, EXT_DOMAINS, num_sep=5)

  if task == 'dup':
    duplicate(UNIFIEDQA_RAW_GS, UNIFIEDQA_RAW_DUP_GS, EXT_DOMAINS, dup_count=10)

  if task == 'first_decode':
    convert_first_token_decoding(UNIFIEDQA_RAW_GS, UNIFIEDQA_RAW_FIRST_DECODE_UQ3B_GS,
                                 EXT_DOMAINS, split='dev', score_file='output/exp/uq_ext/dev/3B/uq.txt')
    convert_first_token_decoding(UNIFIEDQA_RAW_GS, UNIFIEDQA_RAW_FIRST_DECODE_UQ3B_GS,
                                 EXT_DOMAINS, split='train', score_file='output/exp/uq_ext/train/3B/uq.txt')

  if task == 'first_decode_topk':
    convert_first_token_decoding_topk(UNIFIEDQA_RAW_GS, UNIFIEDQA_RAW_FIRST_DECODE_UQ3B_GS, UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_GS,
                                      EXT_DOMAINS, split='dev', score_file='output/exp/uq_ext_first/dev/3B/uq.txt', use_gold=True)
    convert_first_token_decoding_topk(UNIFIEDQA_RAW_GS, UNIFIEDQA_RAW_FIRST_DECODE_UQ3B_GS, UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS,
                                      EXT_DOMAINS, split='dev', score_file='output/exp/uq_ext_first/dev/3B/uq.txt', use_gold=False)
    convert_first_token_decoding_topk(UNIFIEDQA_RAW_GS, UNIFIEDQA_RAW_FIRST_DECODE_UQ3B_GS, UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_GS,
                                      EXT_DOMAINS, split='train', score_file='output/exp/uq_ext_first/train/3B/uq.txt', use_gold=True)
    convert_first_token_decoding_topk(UNIFIEDQA_RAW_GS, UNIFIEDQA_RAW_FIRST_DECODE_UQ3B_GS, UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS,
                                      EXT_DOMAINS, split='train', score_file='output/exp/uq_ext_first/train/3B/uq.txt', use_gold=False)
    multi2one_all(UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_GS, UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_GS_OL, EXT_DOMAINS, num_sep=5)

  if task == 'ext_verify':
    convert_first_token_decoding_topk(UNIFIEDQA_RAW_GS, UNIFIEDQA_RAW_FIRST_DECODE_UQ3B_GS, UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS + '_test',
                                      EXT_DOMAINS, split='dev', score_file='output/exp/uq_ext_first/dev/3B/uq.txt', use_gold=False, num_spans=1)

  if task == 'reducehop_first':
    raw_reducehop_dir = 'data/mh_mh_reducehop_oneline'
    reducehop_first_dir = 'data/mh_mh_reducehop_first_oneline'
    reducehop_first(raw_reducehop_dir, reducehop_first_dir, num_hop=2)

  if task == 'reducehop_second':
    raw_reducehop_dir = 'data/mh_mh_reducehop_oneline'
    reducehop_second_dir = 'data/mh_mh_reducehop_second_oneline'
    prediction_path = 'output/multihop/uq_mh_mh_reducehop_first_ol_mix/dev/11B_mh_mh_reducehop_lr1e3.txt-1200000'
    reducehop_second(raw_reducehop_dir, prediction_path, reducehop_second_dir,
                     num_hop=2, domains=MH_DOMAINS, split='dev')
