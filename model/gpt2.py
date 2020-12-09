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
    t_input = tokenizer(input)
    t_target = tokenizer(target)
    # truncate
    inp = t_input['input_ids'][:max_input_len]
    tar = t_target['input_ids'][:max_target_len - int(add_eos)]
    inp_att = t_input['attention_mask'][:max_input_len]
    tar_att = t_target['attention_mask'][:max_target_len - int(add_eos)]
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
  '''
  assert len(inputs) == len(targets)
  t_inputs = tokenizer(inputs)
  t_targets = tokenizer(targets)
  combine = {k: [] for k in t_inputs}
  split_point = []
  for i in range(len(inputs)):
    inp = t_inputs['input_ids'][i][:max_input_len]
    tar = t_targets['input_ids'][i][:max_target_len - int(add_eos)]
    inp_att = t_inputs['attention_mask'][i][:max_input_len]
    tar_att = t_targets['attention_mask'][i][:max_target_len - int(add_eos)]
    if add_eos:
      tar.append(tokenizer.eos_token_id)
      tar_att.append(1)
    comb_ids = inp + tar
    comb_mask = inp_att + tar_att
    if i == 0 and pad_to_max:
      assert len(comb_ids) <= max_input_len + max_target_len and len(comb_ids) == len(comb_mask)
      comb_ids += [0] * (max_input_len + max_target_len - len(comb_ids))
      comb_mask += [0] * (max_input_len + max_target_len - len(comb_mask))
    combine['input_ids'].append(comb_ids)
    combine['attention_mask'].append(comb_mask)
    split_point.append(len(inp))
  combine['input_ids'] = pad_sequence([torch.LongTensor(i) for i in combine['input_ids']],
                                      batch_first=True, padding_value=0).to(device)
  combine['attention_mask'] = pad_sequence([torch.LongTensor(i) for i in combine['attention_mask']],
                                           batch_first=True, padding_value=0).to(device)
  return combine, split_point
  '''


def compute_logprob(model, input_dict: Dict[str, torch.Tensor]):
  outputs = model(**input_dict, labels=input_dict['input_ids'])
  logits = outputs.logits.detach()
  logprobs = F.log_softmax(logits, dim=-1)
  logprobs = torch.gather(logprobs, -1, input_dict['input_ids'].unsqueeze(-1)).squeeze(-1)
  return logprobs
