import argparse
import matplotlib
import matplotlib.cm as cm
from tqdm import tqdm
import t5
from t5.data.utils import get_default_vocabulary
from dataset import build


def read_score(filename: str, mixture: str, split: str, vocab):
  mix = t5.data.MixtureRegistry.get(mixture)
  ds = mix.get_dataset_in_order(
    split=split, sequence_length={'inputs': 512, 'targets': 512}, shuffle=False)

  with open(filename, 'r') as fin:
    prev_inp = None
    tgt_logprob_li = []
    for l in tqdm(fin):
      try:
        ex = next(ds)
      except StopIteration:
        break
      weight = float(ex['weights'].numpy())
      score, inp, tgt, logprob = l.strip().split('\t')
      score = float(score)
      inp = [int(i) for i in inp.split(',0', 1)[0].split(',')]
      tgt = [int(i) for i in tgt.split(',0', 1)[0].split(',')]
      logprob = [float(i) for i in logprob.split(',')]
      inp = vocab.decode(inp)
      tgt = [vocab.decode([i]) for i in tgt]
      if prev_inp is not None and prev_inp != inp:
        yield prev_inp, tgt_logprob_li
        tgt_logprob_li = []
      tgt_logprob_li.append((tgt, logprob[:len(tgt)], weight, score))
      prev_inp = inp
    if len(tgt_logprob_li) > 0:
      yield inp, tgt_logprob_li


def to_stype(scalar, map, correct: bool) -> str:
  return "style='background-color:rgba({:.0f}%, {:.0f}%, {:.0f}%, {:.0f}%); {}' title='{}'".format(
    *([n * 100 for n in map.to_rgba(scalar)] + (['font-weight:bold' if correct else '']) + [scalar]))


def vis(filename: str, outfilename: str, mixture: str, split: str, vocab, max_count: int=None):
  norm = matplotlib.colors.Normalize(vmin=-70, vmax=0)
  map = cm.ScalarMappable(norm=norm, cmap=cm.hot)
  with open(outfilename, 'w') as fout:
    for i, (inp, tgt_logprob_li) in enumerate(read_score(filename, mixture, split, vocab)):
      if max_count and i >= max_count:
        break
      fout.write('<div><div>{}</div>{}</div><hr/>\n'.format(
        inp, ''.join(['<div>{} {}</div>'.format(
          score, ' '.join(['<span {}>{}</span>'.format(to_stype(l, map, weight==1), t) for t, l in zip(tgt, logprob)]))
          for tgt, logprob, weight, score in tgt_logprob_li])))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='visulize log prob per tokens')
  parser.add_argument('--mix', type=str, help='mixture', default='uq_sub_test_mix')
  parser.add_argument('--split', type=str, help='split', default='dev')
  parser.add_argument('--score', type=str, help='score file')
  parser.add_argument('--out', type=str, help='output file')
  parser.add_argument('--max', type=int, help='max output count', default=None)
  args = parser.parse_args()

  # build tasks and mixtures
  build(neg_method='weight')

  vocab = get_default_vocabulary()
  vis(args.score, args.out, args.mix, args.split, vocab, max_count=args.max)
