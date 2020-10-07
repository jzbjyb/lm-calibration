from typing import List
import argparse
from tqdm import tqdm
import random
import t5
import torch
import torch.nn as nn
import numpy as np
from dataset import build


SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def read_data(filename: str, mixture: str, split: str):
  mix = t5.data.MixtureRegistry.get(mixture)
  ds = mix.get_dataset_in_order(
    split=split, sequence_length={'inputs': 512, 'targets': 512}, shuffle=False)

  with open(filename, 'r') as fin:
    prev_inp = None
    scores_li: List[List[float]] = []
    targets_li: List[List[float]] = []
    scores = []
    targets = []
    for l in tqdm(fin):
      try:
        ex = next(ds)
      except StopIteration:
        break
      weight = float(ex['weights'].numpy())
      inp = ex['inputs_plaintext'].numpy().decode()
      score = l.strip().split('\t', 1)[0]
      score = float(score)
      if prev_inp is not None and prev_inp != inp:
        scores_li.append(scores)
        targets_li.append(targets)
        scores = []
        targets = []
      scores.append(score)
      targets.append(int(weight == 1))
      prev_inp = inp
    if len(scores) > 0:
      scores_li.append(scores)
      targets_li.append(targets)

  return np.array(scores_li), np.array(targets_li)


class TempCal(nn.Module):
  def __init__(self):
    super(TempCal, self).__init__()
    self._temp = nn.Parameter(torch.tensor(0.0))


  @property
  def temp(self):
    return torch.exp(self._temp)


  def forward(self, scores, targets):
    mask = scores.ne(1.0).float()
    logits = (scores / self.temp * targets).sum(-1)
    log_z = torch.logsumexp(scores / self.temp + torch.log(mask), -1)
    log_prob = logits - log_z
    loss = -log_prob.mean()
    return loss


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='visulize log prob per tokens')
  parser.add_argument('--mix', type=str, help='mixture', default='uq_sub_test_mix')
  parser.add_argument('--split', type=str, help='split', default='dev')
  parser.add_argument('--score', type=str, help='score file')
  parser.add_argument('--out', type=str, help='output file')
  args = parser.parse_args()

  # build tasks and mixtures
  build(neg_method='weight')

  scores_li, targets_li = read_data(args.score, args.mix, args.split)
  print('#examples {}'.format(len(scores_li)))


  temp_cal = TempCal()
  temp_cal.train()
  optimizer = torch.optim.Adam(temp_cal.parameters(), lr=1e-1)

  batch_size = 256
  epoch = 10
  num_batch = int(np.ceil(len(scores_li) / batch_size))
  for e in range(epoch):
    loss_li = []
    perm = np.random.permutation(len(scores_li))
    for b in range(num_batch):
      bind = perm[b * batch_size:b * batch_size + batch_size]
      scores = nn.utils.rnn.pad_sequence([torch.tensor(s) for s in scores_li[bind]], batch_first=True, padding_value=1.0)
      targets = nn.utils.rnn.pad_sequence([torch.tensor(t) for t in targets_li[bind]], batch_first=True)
      loss = temp_cal(scores, targets)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss_li.append(loss.detach().cpu().numpy())
    print(np.mean(loss_li), temp_cal.temp)
