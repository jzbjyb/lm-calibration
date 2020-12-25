import argparse
import os
import random
from tqdm import tqdm
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from model.data import read_qa_data, compose_few_shots
from model.gpt2 import string_to_tensor, compute_logprob
CLEAN_TEST_DOMAINS = [('arc_hard', ('train', 'dev', 'test')),
                      ('ai2_science_middle', ('train', 'dev', 'test')),
                      ('mctest_corrected_the_separator', ('train', 'dev')),
                      ('social_iqa', ('train', 'dev')),
                      ('race_string', ('train', 'dev', 'test'))]
CLEAN_TRAIN_DOMAINS = [('arc_easy', ('train', 'dev', 'test')),
                       ('ai2_science_elementary', ('train', 'dev', 'test')),
                       ('openbookqa', ('train', 'dev', 'test')),
                       ('qasc', ('train', 'dev', 'test')),
                       ('winogrande_l', ('train', 'dev')),
                       ('commonsenseqa', ('train', 'dev', 'test')),
                       ('physical_iqa', ('train', 'dev', 'test'))]
SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='score targets using models from transformers')
  parser.add_argument('--model', type=str, help='model name', default='gpt2-large')
  parser.add_argument('--checkpoint', type=str, help='path to the checkpoint', default=None)
  parser.add_argument('--data', type=str, help='path to the data')
  parser.add_argument('--domains', type=str, default='clean_test_domains')
  parser.add_argument('--split', type=str, help='split', default='dev')
  parser.add_argument('--fewshot', type=int, default=0)
  parser.add_argument('--fewshot_data', type=str, help='path to few shot data', default=None)
  parser.add_argument('--fewshot_domains', type=str, default='clean_test_domains')
  parser.add_argument('--output', type=str, help='output file')
  parser.add_argument('--batch_size', type=int, help='batch size', default=0)
  parser.add_argument('--max_token_per_batch', type=int, default=2048)
  parser.add_argument('--has_ret', action='store_true')
  parser.add_argument('--use_inp', action='store_true')
  parser.add_argument('--append_a', action='store_true')
  args = parser.parse_args()
  device = 'cuda'
  max_input_len = 512
  max_target_len = 128

  print('init data')
  fewshot = None
  if args.fewshot:
    data_fs = read_qa_data(args.fewshot_data, eval(args.fewshot_domains.upper()), split='train', has_ret=False, use_inp=False, append_a=False, only_correct=True)
    data_fs = [(q, a) for q, a in data_fs if len(q) <= 256]
    print('totally {} data for few shot sampling'.format(len(data_fs)))
    random.shuffle(data_fs)
    fewshot = compose_few_shots(data_fs[:args.fewshot])
  data = read_qa_data(args.data, eval(args.domains.upper()), split=args.split, has_ret=args.has_ret, use_inp=args.use_inp, append_a=args.append_a, fewshot=fewshot)
  tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
  iter = string_to_tensor(tokenizer, data, max_input_len=max_input_len, max_target_len=max_target_len,
                          add_eos=True, pad_to_max=False, max_token_per_batch=args.max_token_per_batch, device=device)

  print('loading models ...')
  if args.checkpoint:
    model = GPT2LMHeadModel.from_pretrained(args.model, state_dict=torch.load(args.checkpoint)).to(device)
  else:
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device)

  os.makedirs(os.path.dirname(args.output), exist_ok=True)
  pbar = tqdm()
  with open(args.output, 'w') as fout:
    for input_dict, split_point in iter:
      logprobs = compute_logprob(model, input_dict)
      for tokens, lp, sp in zip(input_dict['input_ids'].cpu().numpy(), logprobs.cpu().numpy(), split_point):
        inp_toks, tar_toks = tokens[:sp], tokens[sp:]
        #print(tokenizer.convert_ids_to_tokens(inp_toks), tokenizer.convert_ids_to_tokens(tar_toks))
        tar_lp = lp[sp:]
        score = np.sum(tar_lp * (tar_toks > 0))
        fout.write('{}\t{}\t{}\t{}\n'.format(
          score, ','.join(map(str, inp_toks)), ','.join(map(str, tar_toks)), ','.join(map(str, tar_lp))))
        pbar.update(1)
  pbar.close()
