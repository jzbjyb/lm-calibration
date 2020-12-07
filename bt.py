from typing import List, Tuple, Dict
import os
from collections import defaultdict
from tqdm import tqdm
import torch


TRAIN_DOMAINS = [('arc_easy', ('train', 'dev', 'test')),
                 ('ai2_science_elementary', ('train', 'dev', 'test')),
                 ('openbookqa', ('train', 'dev', 'test')),
                 ('qasc', ('train', 'dev', 'test')),
                 ('winogrande_l', ('train', 'dev')),
                 ('commonsenseqa', ('train', 'dev', 'test'))]
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
EXT_DOMAINS = [('squad1_1', ('train', 'dev')),
               ('squad2', ('train', 'dev')),
               ('newsqa', ('train', 'dev')),
               ('quoref', ('train', 'dev')),
               ('ropes', ('train', 'dev'))]
DOMAINS = TRAIN_DOMAINS + TEST_DOMAINS


def qa_dataset_backtranslate(from_file: str,
                             to_file: str,
                             trans1=None,
                             trans2=None,
                             bt_count: int=1,
                             out_count: int=1):
  assert bt_count >= out_count
  out_of_bound_count = all_count = 0
  os.makedirs(os.path.dirname(to_file), exist_ok=True)
  with open(from_file, 'r') as fin, open(to_file, 'w') as fout, open(to_file + '.raw', 'w') as fout_raw:
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

      tok2_bins = trans1.generate(tok1_bin, beam=bt_count)
      tok2_bins = [tok2_bin['tokens'].cpu() for tok2_bin in tok2_bins]

      bts: Dict[str, int] = defaultdict(lambda: 0)
      bts_li: List[str] = []
      tok1_binss = trans2.generate(tok2_bins, beam=bt_count)
      for tok1_bins in tok1_binss:
        for tok1_bin in tok1_bins:
          tok1_bpe = trans2.string(tok1_bin['tokens'])
          tok1 = trans2.remove_bpe(tok1_bpe)
          an = trans2.detokenize(tok1)
          bts[an] += 1
          bts_li.append(an)
          fout_raw.write('{}\t{}\t{}\t{}\n'.format('{}-{}'.format(qid, ans_ind), question, an, correct))

      # remove the original sentence if possible
      if answer in bts:
        bts[answer] -= (bt_count * bt_count - out_count)
        if bts[answer] <= 0:
          del bts[answer]

      bts = sorted(bts.items(), key=lambda x: -x[1])
      c = out_count
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
        fout.write('{}\t{}\t{}\t{}\n'.format('{}-{}'.format(qid, ans_ind), question, bts[ind][0], correct))
        bts[ind] = (bts[ind][0], bts[ind][1] - 1)
        ind += 1
        c -= 1

      prev_qid = qid
  return out_of_bound_count, all_count


def bt(from_dir, to_dir, domains: List[Tuple[str, Tuple]], format: str='tsv', restricted_splits=None, **kwargs):
  en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
  en2de.cuda()
  print('loaded en2de')
  de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')
  de2en.cuda()
  print('loaded de2en')
  for domain, splits in domains:
    for split in splits:
      if restricted_splits is not None and split not in restricted_splits:
        continue
      in_fname = os.path.join(from_dir, domain, split + '.' + format)
      out_fname = os.path.join(to_dir, domain, split + '.' + format)
      ooc, ac = qa_dataset_backtranslate(in_fname, out_fname, trans1=en2de, trans2=de2en, **kwargs)
      print('{} -> {}, {} out of bound from {}'.format(in_fname, out_fname, ooc, ac), flush=True)


bt('data/unifiedqa', 'data/unifiedqa_bt_dedup', TEST_DOMAINS, bt_count=10, out_count=4, restricted_splits={'dev'})
bt('data/unifiedqa', 'data/unifiedqa_bt_dedup', TRAIN_DOMAINS, bt_count=10, out_count=4, restricted_splits={'dev'})
bt('data/test_prep', 'data/test_prep_bt_dedup', [('', ('test',))], bt_count=10, out_count=4, restricted_splits={'test'})
bt('data/unifiedqa_decode_uq3B', 'data/unifiedqa_decode_uq3B_bt_dedup', EXT_DOMAINS, bt_count=10, out_count=4, restricted_splits={'dev'})
bt('data/unifiedqa_decode_uq3B_dedup', 'data/unifiedqa_decode_uq3B_dedup_bt_dedup', EXT_DOMAINS, bt_count=10, out_count=4, restricted_splits={'dev'})
bt('data/unifiedqa_decode_uq3B_sample', 'data/unifiedqa_decode_uq3B_sample_bt_dedup', EXT_DOMAINS, bt_count=10, out_count=4, restricted_splits={'dev'})
bt('data/unifiedqa_decode_uq3B_span_topk_nogold', 'data/unifiedqa_decode_uq3B_span_topk_nogold_bt_dedup', EXT_DOMAINS, bt_count=10, out_count=4, restricted_splits={'dev'})
