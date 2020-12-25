from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def string_to_tensor(tokenizer,
                     data,
                     max_input_len: int,
                     max_target_len: int,
                     add_eos: bool=True,
                     pad_to_max: bool=False,
                     max_token_per_batch: int=None,
                     device: str='cuda') -> Tuple[Dict[str, torch.Tensor], List[int]]:
  combine = {'input_ids': [], 'attention_mask': []}
  split_point = []
  max_len_per_example: int = 0
  for input, target in data:
    # tokenize
    t_input = tokenizer.batch_encode_plus([input])
    t_target = tokenizer.batch_encode_plus([target])
    # truncate
    inp = t_input['input_ids'][0][:max_input_len]
    tar = t_target['input_ids'][0][:max_target_len - int(add_eos)]
    inp_att = t_input['attention_mask'][0][:max_input_len]
    tar_att = t_target['attention_mask'][0][:max_target_len - int(add_eos)]
    # add eos
    if add_eos:
      tar.append(tokenizer.eos_token_id)
      tar_att.append(1)
    # combine
    comb_ids = inp + tar
    comb_mask = inp_att + tar_att
    # padding
    if pad_to_max:
      assert len(comb_ids) <= max_input_len + max_target_len and len(comb_ids) == len(comb_mask)
      comb_ids += [0] * (max_input_len + max_target_len - len(comb_ids))
      comb_mask += [0] * (max_input_len + max_target_len - len(comb_mask))
    # yield or not
    cur_len = len(comb_ids)
    if max(cur_len, max_len_per_example) * (len(combine['input_ids']) + 1) > max_token_per_batch:
      combine['input_ids'] = pad_sequence([torch.LongTensor(i) for i in combine['input_ids']],
                                          batch_first=True, padding_value=0).to(device)
      combine['attention_mask'] = pad_sequence([torch.LongTensor(i) for i in combine['attention_mask']],
                                               batch_first=True, padding_value=0).to(device)
      yield combine, split_point
      combine = {'input_ids': [], 'attention_mask': []}
      split_point = []
      max_len_per_example = 0
    # save
    combine['input_ids'].append(comb_ids)
    combine['attention_mask'].append(comb_mask)
    split_point.append(len(inp))
    max_len_per_example = max(max_len_per_example, cur_len)
  if len(combine['input_ids']) > 0:
    combine['input_ids'] = pad_sequence([torch.LongTensor(i) for i in combine['input_ids']],
                                        batch_first=True, padding_value=0).to(device)
    combine['attention_mask'] = pad_sequence([torch.LongTensor(i) for i in combine['attention_mask']],
                                             batch_first=True, padding_value=0).to(device)
    yield combine, split_point


def compute_logprob(model, input_dict: Dict[str, torch.Tensor]):  # first token is omitted
  labels = input_dict['input_ids']
  outputs = model(**input_dict, labels=labels)
  logits = outputs[1].detach()
  logprobs = F.log_softmax(logits, dim=-1)
  #logprobs = torch.gather(logprobs, -1, input_dict['input_ids'].unsqueeze(-1)).squeeze(-1)
  logprobs = torch.gather(logprobs[:, :-1, :], -1, labels[:, 1:].unsqueeze(-1)).squeeze(-1)
  first_lps = torch.zeros_like(logprobs[:, :1])  # assume the first token always have prob of 1
  logprobs = torch.cat([first_lps, logprobs], 1)
  return logprobs
