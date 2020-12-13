from typing import List, Tuple
import os


def compose_few_shots(fewshot: List[Tuple[str, str]]=[]):
  qas = []
  for q, a in fewshot:
    q = q.replace('\\n', '\n')
    qas.append('{} = {}'.format(q, a))
  return '\n'.join(qas)


def read_qa_data(data_dir: str, domains: List, split: str, format: str='tsv', has_ret: bool=False, use_inp: bool=False, append_a: bool=False, fewshot: str=None, only_correct: bool=False):
  for domain, splits in domains:
    file = os.path.join(data_dir, domain, split + '.' + format)
    with open(file, 'r') as fin:
      for l in fin:
        if has_ret:
          ls = l.rstrip('\n').split('\t')
          lid, question, answer, correct = ls[:4]
          if only_correct and correct != 'True':
            continue
          question = question.replace('\\n', '\n')
          rets = ls[4:]
          qrets = rets[:len(rets) // 2]
          question = question + ' \n ' + qrets[0]
          if append_a:
            question += ' A:'
          elif fewshot:
            question = fewshot + '\n' + question + ' = '
          if use_inp:
            yield '', question
          else:
            yield question, answer
        else:
          ls = l.rstrip('\n').split('\t')
          if len(ls) == 4:
            lid, question, answer, correct = ls
            if only_correct and correct != 'True':
              continue
          elif len(ls) == 2:
            question, answer = ls
          else:
            raise NotImplementedError
          question = question.replace('\\n', '\n')
          if append_a:
            question += ' A:'
          elif fewshot:
            question = fewshot + '\n' + question + ' = '
          if use_inp:
            yield '', question
          else:
            yield question, answer
