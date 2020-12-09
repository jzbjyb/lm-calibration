import argparse
import os
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from model.data import read_qa_data
from model.gpt2 import string_to_tensor, compute_logprob
CLEAN_TEST_DOMAINS = [('arc_hard', ('train', 'dev', 'test')),
                      ('ai2_science_middle', ('train', 'dev', 'test')),
                      ('mctest_corrected_the_separator', ('train', 'dev')),
                      ('social_iqa', ('train', 'dev')),
                      ('race_string', ('train', 'dev', 'test'))]


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='score targets using models from transformers')
  parser.add_argument('--model', type=str, help='model name', default='gpt2')
  parser.add_argument('--data', type=str, help='path to the data')
  parser.add_argument('--split', type=str, help='split')
  parser.add_argument('--output', type=str, help='output file')
  parser.add_argument('--batch_size', type=int, help='batch size', default=32)
  parser.add_argument('--has_ret', action='store_true')
  parser.add_argument('--use_inp', action='store_true')
  args = parser.parse_args()
  device = 'cuda'
  max_input_len = 512
  max_target_len = 128

  print('loading models ...')
  model = GPT2LMHeadModel.from_pretrained(args.model).to(device)
  tokenizer = GPT2TokenizerFast.from_pretrained(args.model)

  data = read_qa_data(args.data, CLEAN_TEST_DOMAINS, split='dev', has_ret=args.has_ret, use_inp=args.use_inp)

  os.makedirs(os.path.dirname(args.output), exist_ok=True)
  with open(args.output, 'w') as fout:
    while True:
      inputs, targets = list(zip(*[next(data) for i in range(args.batch_size)]))
      inputs, targets = list(inputs), list(targets)
      input_dict, split_point = string_to_tensor(tokenizer, inputs, targets, max_input_len=max_input_len, max_target_len=max_target_len, add_eos=True, device=device)
      logprobs = compute_logprob(model, input_dict)
      for inp, tar, tokens, lp, sp in zip(inputs, targets, input_dict['input_ids'].cpu().numpy(), logprobs.cpu().numpy(), split_point):
        inp_toks, tar_toks = tokens[:sp], tokens[sp:]
        tar_lp = lp[sp:]
        score = np.sum(tar_lp * (tar_toks > 0))
        fout.write('{}\t{}\t{}\t{}\n'.format(
          score, ','.join(map(str, inp_toks)), ','.join(map(str, tar_toks)), ','.join(map(str, tar_lp))))
