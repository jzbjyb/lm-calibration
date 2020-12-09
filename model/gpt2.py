from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def string_to_tensor(tokenizer,
                     inputs: List[str],
                     targets: List[str],
                     max_input_len: int,
                     max_target_len: int,
                     add_eos: bool=True,
                     device: str='cuda') -> Tuple[Dict[str, torch.Tensor], List[int]]:
  assert len(inputs) == len(targets)
  t_inputs = tokenizer(inputs)
  t_targets = tokenizer(targets)
  combine = {k: [] for k in t_inputs}
  split_point = []
  for i in range(len(inputs)):
    inp = t_inputs['input_ids'][i][:max_input_len]
    tar = t_targets['input_ids'][i][:max_target_len]
    inp_att = t_inputs['attention_mask'][i][:max_input_len]
    tar_att = t_targets['attention_mask'][i][:max_target_len]
    if add_eos:
      tar.append(tokenizer.eos_token_id)
      tar_att.append(1)
    combine['input_ids'].append(inp + tar)
    combine['attention_mask'].append(inp_att + tar_att)
    split_point.append(len(inp))
  combine['input_ids'] = pad_sequence([torch.LongTensor(i) for i in combine['input_ids']],
                                      batch_first=True, padding_value=0).to(device)
  combine['attention_mask'] = pad_sequence([torch.LongTensor(i) for i in combine['attention_mask']],
                                           batch_first=True, padding_value=0).to(device)
  return combine, split_point


def compute_logprob(model, input_dict: Dict[str, torch.Tensor]):
  outputs = model(**input_dict, labels=input_dict['input_ids'])
  logits = outputs.logits.detach()
  logprobs = F.log_softmax(logits, dim=-1)
  logprobs = torch.gather(logprobs, -1, input_dict['input_ids'].unsqueeze(-1)).squeeze(-1)
  return logprobs
