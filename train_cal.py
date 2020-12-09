from typing import List
import argparse
import random
import os
import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
from dataset import build, read_score_data, convert_data_to_dmatrix


SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def read_data(filename: str, mixture: str, split: str, **kwargs):
  scores_li: List[List[float]] = []
  input_len_li: List[List[int]] = []
  target_len_li: List[List[int]] = []
  score_var_li: List[List[float]] = []
  targets_li: List[List[float]] = []
  inp_perp_li: List[List[float]] = []

  for sample in read_score_data(filename, mixture, split, **kwargs):
    if np.sum(sample['target']) <= 0:  # skip examples without gold
      continue
    scores_li.append(sample['log_prob'])
    score_var_li.append(sample['prob_var'])
    input_len_li.append(sample['input_len'])
    target_len_li.append(sample['target_len'])
    targets_li.append(sample['target'])
    if 'inp_perp' in sample:
      inp_perp_li.append(sample['inp_perp'])

  data = {'log_prob': np.array(scores_li),
          'prob_var': np.array(score_var_li),
          'input_len': np.array(input_len_li),
          'target_len': np.array(target_len_li),
          'target': np.array(targets_li)}
  if len(inp_perp_li) > 0:
    data['inp_perp'] = np.array(inp_perp_li)
  return data


class TempCal(nn.Module):
  def __init__(self):
    super(TempCal, self).__init__()
    self._temp = nn.Parameter(torch.tensor(0.0))


  @property
  def temp(self):
    return torch.exp(self._temp)


  def forward(self, scores, targets):
    mask = scores.ne(1.0).float()
    logits = torch.logsumexp(scores / self.temp + torch.log(targets.float()), -1)
    log_z = torch.logsumexp(scores / self.temp + torch.log(mask), -1)
    log_prob = logits - log_z
    loss = -log_prob.mean()
    return loss


def train_temp(args, data):
  temp_cal = TempCal()
  temp_cal.train()
  optimizer = torch.optim.Adam(temp_cal.parameters(), lr=1e-3)

  scores_li = data['log_prob']
  targets_li = data['target']

  batch_size = 256
  epoch = 300
  early_stop = 30
  es = 0
  num_batch = int(np.ceil(len(scores_li) / batch_size))
  min_loss = 1e10
  min_temp = None
  for e in range(epoch):
    loss_li = []
    perm = np.random.permutation(len(scores_li))
    for b in range(num_batch):
      bind = perm[b * batch_size:b * batch_size + batch_size]
      scores = nn.utils.rnn.pad_sequence([torch.tensor(s) for s in scores_li[bind]], batch_first=True,
                                         padding_value=1.0)
      targets = nn.utils.rnn.pad_sequence([torch.tensor(t) for t in targets_li[bind]], batch_first=True)
      loss = temp_cal(scores, targets)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss_li.append(loss.detach().cpu().numpy())
    loss = np.mean(loss_li)
    if loss < min_loss:
      min_loss = loss
      min_temp = temp_cal.temp.detach().cpu().numpy()
      es = 0
    else:
      es += 1
      if es >= early_stop:
        print('early stop')
        break
    print(e, loss, temp_cal.temp)
  print('final loss {}, final temp {}'.format(min_loss, min_temp))


def train_xgb(args, data):
  dm_train, dm_dev = convert_data_to_dmatrix(data, split=0.8)
  print('#train {}, #dev {}'.format(dm_train.num_row(), dm_dev.num_row()))
  param = {'max_depth': 4, 'subsample': 0.8, 'num_parallel_tree': 5, 'objective': 'binary:logistic'}
  evals = [(dm_train, 'train'), (dm_dev, 'eval')]
  num_round = 100
  bst = xgb.train(param, dm_train, num_round, evals=evals, early_stopping_rounds=5)
  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  bst.save_model(args.out)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='visulize log prob per tokens')
  parser.add_argument('--model', type=str, help='model to train', choices=['xgb', 'temp'])
  parser.add_argument('--mix', type=str, help='mixture', default='uq_sub_test_mix')
  parser.add_argument('--split', type=str, help='split', default='dev')
  parser.add_argument('--score', type=str, help='score file')
  parser.add_argument('--inp_perp', type=str, help='feature of input perplexity', default=None)
  parser.add_argument('--out', type=str, help='output file')
  args = parser.parse_args()

  # build tasks and mixtures
  build(neg_method='weight')

  data = read_data(args.score, args.mix, args.split, inp_perp=args.inp_perp)
  print('#examples {}'.format(len(data['target'])))

  if args.model == 'temp':
    train_temp(args, data)
  elif args.model == 'xgb':
    train_xgb(args, data)
