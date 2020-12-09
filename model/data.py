from typing import List
import os


def read_qa_data(data_dir: str, domains: List, split: str, format: str='tsv', has_ret: bool=False, use_inp: bool=False, append_a: bool=False):
  for domain, splits in domains:
    file = os.path.join(data_dir, domain, split + '.' + format)
    with open(file, 'r') as fin:
      for l in fin:
        if has_ret:
          ls = l.rstrip('\n').split('\t')
          lid, question, answer, correct = ls[:4]
          question = question.replace('\\n', '\n')
          rets = ls[4:]
          qrets = rets[:len(rets) // 2]
          question = question + ' \n ' + qrets[0]
          if append_a:
            question += ' A:'
          if use_inp:
            yield '', question
          else:
            yield question, answer
        else:
          lid, question, answer, correct = l.rstrip('\n').split('\t')
          question = question.replace('\\n', '\n')
          if append_a:
            question += ' A:'
          if use_inp:
            yield '', question
          else:
            yield question, answer
