from typing import List, Tuple
import os
from tqdm import tqdm
import torch


TRAIN_DOMAINS = [('arc_easy', ('train', 'dev', 'test')),
                 ('ai2_science_elementary', ('train', 'dev', 'test')),
                 ('openbookqa', ('train', 'dev', 'test')),
                 ('qasc', ('train', 'dev')),
                 ('winogrande_l', ('train', 'dev')),
                 ('commonsenseqa', ('train', 'dev'))]
TEST_DOMAINS = [('arc_hard', ('train', 'dev', 'test')),
                ('ai2_science_middle', ('train', 'dev', 'test')),
                ('winogrande_m', ('train', 'dev')),
                ('winogrande_s', ('train', 'dev')),
                ('mctest_corrected_the_separator', ('train', 'dev')),
                ('physical_iqa', ('train', 'dev', 'test')),
                ('social_iqa', ('train', 'dev')),
                ('race_string', ('train', 'dev', 'test'))]
SUB_TEST_DOMAINS = [('arc_hard', ('train', 'dev', 'test')),
                    ('ai2_science_middle', ('train', 'dev', 'test')),
                    ('winogrande_m', ('train', 'dev')),
                    ('winogrande_s', ('train', 'dev')),
                    ('mctest_corrected_the_separator', ('train', 'dev'))]
DOMAINS = TRAIN_DOMAINS + TEST_DOMAINS


def qa_dataset_backtranslate(from_file: str,
                             to_file: str,
                             trans1=None,
                             trans2=None,
                             bt_count: int=1):
  os.makedirs(os.path.dirname(to_file), exist_ok=True)
  with open(from_file, 'r') as fin, open(to_file, 'w') as fout:
    ans_ind = 0
    prev_qid = None
    for lid, l in tqdm(enumerate(fin)):
      qid, question, answer, correct = l.strip().split('\t')
      if qid != prev_qid:
        ans_ind = 0
      else:
        ans_ind += 1
      fout.write('{}\t{}\t{}\t{}\n'.format('{}-{}'.format(qid, ans_ind), question, answer, correct))
      tok1 = trans1.tokenize(answer)
      tok1_bpe = trans1.apply_bpe(tok1)
      tok1_bin = trans1.binarize(tok1_bpe)

      tok2_bins = trans1.generate(tok1_bin, beam=bt_count, sampling=True, sampling_topk=20)
      for tok2_bin in tok2_bins:
        tok1_bins = trans2.generate(tok2_bin['tokens'].cpu(), beam=bt_count, sampling=True, sampling_topk=20)
        for tok1_bin in tok1_bins:
          tok1_bpe = trans2.string(tok1_bin['tokens'])
          tok1 = trans2.remove_bpe(tok1_bpe)
          an = trans2.detokenize(tok1)
          fout.write('{}\t{}\t{}\t{}\n'.format('{}-{}'.format(qid, ans_ind), question, an, correct))

      prev_qid = qid


def bt(from_dir, to_dir, domains: List[Tuple[str, List[str]]], format: str='tsv', **kwargs):
  en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
  en2de.cuda()
  print('loaded en2de')
  de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')
  de2en.cuda()
  print('loaded de2en')
  for domain, splits in domains:
    for split in splits:
      if split != 'dev':
        continue
      in_fname = os.path.join(from_dir, domain, split + '.' + format)
      out_fname = os.path.join(to_dir, domain, split + '.' + format)
      print('{} -> {}'.format(in_fname, out_fname))
      qa_dataset_backtranslate(in_fname, out_fname, trans1=en2de, trans2=de2en, **kwargs)
      break


bt('data/unifiedqa', 'data/unifiedqa_bt', SUB_TEST_DOMAINS, bt_count=2)
