from typing import Tuple, Union, List, Set
import functools
import tensorflow as tf
import t5
import gin
from .utils import trivia_preprocessor, qa_dataset_fn, qa_dataset_fn_oneline, \
  qa_dataset_fn_onlycorrect, concat_preprocessor, qa_dataset_fn_ret, qa_dataset_onlyinput_fn, \
  qa_dataset_fn_onlycorrect_multi


UNIFIEDQA_GS = 'gs://unifiedqa/data'
UNIFIEDQA_PREP_GS = 'gs://neulab-qa/data/unifiedqa'
UNIFIEDQA_PREP_GS_RET_DRQA = 'gs://neulab-qa/data/unifiedqa_ret_drqa'
UNIFIEDQA_PREP_GS_RET_DRQA_3S = 'gs://neulab-qa/data/unifiedqa_ret_drqa_3s'
UNIFIEDQA_RAW_GS = 'gs://neulab-qa/unifiedqa/data'
UNIFIEDQA_RAW_DUP_GS = 'gs://neulab-qa/data/unifiedqa_dup'
UNIFIEDQA_PREP_GS_OL = 'gs://neulab-qa/data/unifiedqa_oneline'
UNIFIEDQA_PREP_GS_BT = 'gs://neulab-qa/data/unifiedqa_bt'
UNIFIEDQA_PREP_GS_BT_DEDUP = 'gs://neulab-qa/data/unifiedqa_bt_dedup'
UNIFIEDQA_PREP_GS_BT_DEDUP_TOP10 = 'gs://neulab-qa/data/unifiedqa_bt_dedup_top10'
UNIFIEDQA_PREP_GS_BT_DEDUP_TOP20 = 'gs://neulab-qa/data/unifiedqa_bt_dedup_top20'
UNIFIEDQA_PREP_GS_BT_REP = 'gs://neulab-qa/data/unifiedqa_bt_replace'
UNIFIEDQA_PREP_GS_BT_DEDUP_REP = 'gs://neulab-qa/data/unifiedqa_bt_dedup_replace'
UNIFIEDQA_PREP_GS_BT_DEDUP_TOP10_REP = 'gs://neulab-qa/data/unifiedqa_bt_dedup_top10_replace'
UNIFIEDQA_PREP_GS_BT_DEDUP_TOP20_REP = 'gs://neulab-qa/data/unifiedqa_bt_dedup_top20_replace'
UNIFIEDQA_PREP_GS_RET_DRQA_3S_BT_REP = 'gs://neulab-qa/data/unifiedqa_ret_drqa_3s_bt_replace'
UNIFIEDQA_PREP_GS_RET_DRQA_3S_BT_DEDUP_REP = 'gs://neulab-qa/data/unifiedqa_ret_drqa_3s_bt_dedup_replace'

UNIFIEDQA_MH_GS_OL = 'gs://neulab-qa/data/mh_oneline'
UNIFIEDQA_MH_DEV_GS_OL = 'gs://neulab-qa/data/mh_dev_oneline'
UNIFIEDQA_MH_MULTIHOP_GS_OL = 'gs://neulab-qa/data/mh_mh_oneline'
UNIFIEDQA_MH_MULTIHOP_SQL_SIMPLE_GS_OL = 'gs://neulab-qa/data/mh_mh_sql_simple_oneline'
UNIFIEDQA_MH_MULTIHOP_SQL_SIMPLE_COMBINE_GS_OL = 'gs://neulab-qa/data/mh_mh_sql_simple_combine_oneline'
UNIFIEDQA_MH_MULTIHOP_SQL_GS_OL = 'gs://neulab-qa/data/mh_mh_sql_oneline'
UNIFIEDQA_MH_MULTIHOP_SQL_COMBINE_GS_OL = 'gs://neulab-qa/data/mh_mh_sql_combine_oneline'
UNIFIEDQA_MH_MULTIHOP_SQL_COMBINE_NOMH_GS_OL = 'gs://neulab-qa/data/mh_mh_sql_combine_nomh_oneline'
UNIFIEDQA_MH_MULTIHOP_SQL_COMBINE_NONLMH_GS_OL = 'gs://neulab-qa/data/mh_mh_sql_combine_nonlmh_oneline'
UNIFIEDQA_MH_MULTIHOP_SQL_COMBINE_CONCAT_GS_OL = 'gs://neulab-qa/data/mh_mh_sql_combine_concat_as_mh_oneline'
UNIFIEDQA_MH_MULTIHOP_CWQ_ELQ_GS_OL = 'gs://neulab-qa/data/mh_mh_cwq_elq_oneline'
UNIFIEDQA_MH_MULTIHOP_DEDUP_GS_OL = 'gs://neulab-qa/data/mh_mh_dedup_oneline'
UNIFIEDQA_MH_MULTIHOP_PATH_GS_OL = 'gs://neulab-qa/data/mh_mh_path_oneline'
UNIFIEDQA_MH_MULTIHOP_PATH_INVERSE_GS_OL = 'gs://neulab-qa/data/mh_mh_path_inverse_oneline'
UNIFIEDQA_MH_MULTIHOP_HINT_GS_OL = 'gs://neulab-qa/data/mh_mh_hint_oneline'
UNIFIEDQA_MH__SM_GS_OL = 'gs://neulab-qa/data/mh_-sm_oneline'
UNIFIEDQA_MH_S_M_GS_OL = 'gs://neulab-qa/data/mh_s-m_oneline'
UNIFIEDQA_MH_MULTIHOP_REDUCEHOP_GS_OL = 'gs://neulab-qa/data/mh_mh_reducehop_oneline'
UNIFIEDQA_MH_MULTIHOP_REDUCEHOP_FIRST_GS_OL = 'gs://neulab-qa/data/mh_mh_reducehop_first_oneline'
UNIFIEDQA_MH_MULTIHOP_REDUCEHOP_SECOND_GS_OL = 'gs://neulab-qa/data/mh_mh_reducehop_second_oneline'
UNIFIEDQA_MH_MULTIHOP_IMPLICIT_GS_OL = 'gs://neulab-qa/data/mh_mh_implicit_oneline'
UNIFIEDQA_MH_MULTIHOP_IMPLICIT_NOMH_GS_OL = 'gs://neulab-qa/data/mh_mh_implicit_nomh_oneline'
UNIFIEDQA_MH_MULTIHOP_CWQ_ELQ_IMPLICIT_NOMH_GS_OL = 'gs://neulab-qa/data/mh_mh_cwq_elq_implicit_nomh_oneline'
UNIFIEDQA_MH_MULTIHOP_EXPLICIT_GS_OL = 'gs://neulab-qa/data/mh_mh_explicit_oneline'
UNIFIEDQA_MH_FIRST_GS_OL = 'gs://neulab-qa/data/mh_first_oneline'
UNIFIEDQA_MH_FIRST_HINT_GS_OL = 'gs://neulab-qa/data/mh_first_hint_oneline'
UNIFIEDQA_MH_SECOND_GS_OL = 'gs://neulab-qa/data/mh_second_oneline'
UNIFIEDQA_MH_SECOND_PATH_GS_OL = 'gs://neulab-qa/data/mh_second_path_oneline'
UNIFIEDQA_MH_SECOND_PATH_INVERSE_GS_OL = 'gs://neulab-qa/data/mh_second_path_inverse_oneline'
UNIFIEDQA_MH_SECOND_ALLTHEWAY_GS_OL = 'gs://neulab-qa/data/mh_second_alltheway_oneline'
UNIFIEDQA_MH_SECOND_HINT_GS_OL = 'gs://neulab-qa/data/mh_second_hint_oneline'
UNIFIEDQA_MH_MULTIHOP_STATEMENT_GS_OL = 'gs://neulab-qa/data/mh_mh_statement_oneline'

UNIFIEDQA_MH_TOY_GS_OL = 'gs://neulab-qa/data/toy'
UNIFIEDQA_MH_TOY_BAL_GS_OL = 'gs://neulab-qa/data/toy_balanced'

UNIFIEDQA_RAW_DECODE_GS = 'gs://neulab-qa/data/unifiedqa_decode'
UNIFIEDQA_RAW_DECODE_GS_ANS = 'gs://neulab-qa/data/unifiedqa_decode_ans'
UNIFIEDQA_RAW_DECODE_GS_ANS_NO = 'gs://neulab-qa/data/unifiedqa_decode_ans_no'
UNIFIEDQA_RAW_DECODE_GS_OL = 'gs://neulab-qa/data/unifiedqa_decode_ol'
UNIFIEDQA_RAW_DECODE_GS_OL_ANS = 'gs://neulab-qa/data/unifiedqa_decode_ol_ans'
UNIFIEDQA_RAW_DECODE_GS_OL_ANS_NO = 'gs://neulab-qa/data/unifiedqa_decode_ol_ans_no'

UNIFIEDQA_RAW_DECODE_UQ3B_GS = 'gs://neulab-qa/data/unifiedqa_decode_uq3B'
UNIFIEDQA_RAW_DECODE_UQ3B_GS_OL = 'gs://neulab-qa/data/unifiedqa_decode_ol_uq3B'
UNIFIEDQA_RAW_DECODE_UQ3B_GS_RET_DRQA = 'gs://neulab-qa/data/unifiedqa_decode_uq3B_ret_drqa'
UNIFIEDQA_RAW_DECODE_UQ3B_GS_RET_DRQA_3S = 'gs://neulab-qa/data/unifiedqa_decode_uq3B_ret_drqa_3s'
UNIFIEDQA_RAW_DECODE_UQ3B_GS_BT = 'gs://neulab-qa/data/unifiedqa_decode_uq3B_bt'
UNIFIEDQA_RAW_DECODE_UQ3B_GS_RET_DRQA_3S_BT = 'gs://neulab-qa/data/unifiedqa_decode_uq3B_ret_drqa_3s_ret'

UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS = 'gs://neulab-qa/data/unifiedqa_decode_uq3B_dedup'
UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_OL = 'gs://neulab-qa/data/unifiedqa_decode_ol_uq3B_dedup'
UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_RET_DRQA = 'gs://neulab-qa/data/unifiedqa_decode_uq3B_dedup_ret_drqa'
UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_RET_DRQA_3S = 'gs://neulab-qa/data/unifiedqa_decode_uq3B_dedup_ret_drqa_3s'
UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_BT = 'gs://neulab-qa/data/unifiedqa_decode_uq3B_dedup_bt'
UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_RET_DRQA_3S_BT = 'gs://neulab-qa/data/unifiedqa_decode_uq3B_dedup_ret_drqa_3s_ret'

UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS = 'gs://neulab-qa/data/unifiedqa_decode_uq3B_sample'
UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_OL = 'gs://neulab-qa/data/unifiedqa_decode_ol_uq3B_sample'
UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_RET_DRQA = 'gs://neulab-qa/data/unifiedqa_decode_uq3B_sample_ret_drqa'
UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_RET_DRQA_3S = 'gs://neulab-qa/data/unifiedqa_decode_uq3B_sample_ret_drqa_3s'
UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_BT = 'gs://neulab-qa/data/unifiedqa_decode_uq3B_sample_bt'
UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_RET_DRQA_3S_BT = 'gs://neulab-qa/data/unifiedqa_decode_uq3B_sample_ret_drqa_3s_ret'

UNIFIEDQA_RAW_FIRST_DECODE_UQ3B_GS = 'gs://neulab-qa/data/unifiedqa_first_decode_uq3B'
UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_GS = 'gs://neulab-qa/data/unifiedqa_decode_uq3B_span_topk'
UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_GS_OL = 'gs://neulab-qa/data/unifiedqa_decode_ol_uq3B_span_topk'
UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS = 'gs://neulab-qa/data/unifiedqa_decode_uq3B_span_topk_nogold'
UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS_BT = 'gs://neulab-qa/data/unifiedqa_decode_uq3B_span_topk_nogold_bt_dedup'
UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS_RET_DRQA = 'gs://neulab-qa/data/unifiedqa_decode_uq3B_span_topk_nogold_ret_drqa'
UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS_RET_DRQA_3S = 'gs://neulab-qa/data/unifiedqa_decode_uq3B_span_topk_nogold_ret_drqa_3s'
UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS_RET_DRQA_3S_BT = 'gs://neulab-qa/data/unifiedqa_decode_uq3B_span_topk_nogold_ret_drqa_3s_bt_dedup'

#MH_DOMAINS = [('hotpotqa', ('train', 'dev')),
#              ('complexwebq', ('train', 'dev'))]
MH_DOMAINS = [('complexwebq', ('train', 'dev'))]
HP_MH_DOMAINS = [('hotpotqa', ('train', 'dev'))]
TOY_DOMAINS = [('trex', ('test'))]

TRAIN_DOMAINS = [('arc_easy', ('train', 'dev', 'test')),
                 ('ai2_science_elementary', ('train', 'dev', 'test')),
                 ('openbookqa', ('train', 'dev', 'test')),
                 ('qasc', ('train', 'dev', 'test')),
                 ('winogrande_l', ('train', 'dev')),
                 ('commonsenseqa', ('train', 'dev', 'test'))]
TEST_DOMAINS = [('arc_hard', ('train', 'dev', 'test')),
                ('ai2_science_middle', ('train', 'dev', 'test')),
                ('winogrande_m', ('train', 'dev')),
                ('winogrande_s', ('train', 'dev')),
                ('mctest_corrected_the_separator', ('train', 'dev')),
                ('physical_iqa', ('train', 'dev', 'test')),
                ('social_iqa', ('train', 'dev')),
                ('race_string', ('train', 'dev', 'test'))]
SUB_TEST_DOMAINS = [('arc_hard', ('train', 'dev', 'test')),
                    ('ai2_science_middle', ('train', 'dev', 'test')),
                    ('winogrande_m', ('train', 'dev')),
                    ('winogrande_s', ('train', 'dev')),
                    ('mctest_corrected_the_separator', ('train', 'dev'))]
CLEAN_TRAIN_DOMAINS = [('arc_easy', ('train', 'dev', 'test')),
                       ('ai2_science_elementary', ('train', 'dev', 'test')),
                       ('openbookqa', ('train', 'dev', 'test')),
                       ('qasc', ('train', 'dev', 'test')),
                       ('winogrande_l', ('train', 'dev')),
                       ('commonsenseqa', ('train', 'dev', 'test')),
                       ('physical_iqa', ('train', 'dev', 'test'))]
CLEAN_TEST_DOMAINS = [('arc_hard', ('train', 'dev', 'test')),
                      ('ai2_science_middle', ('train', 'dev', 'test')),
                      ('mctest_corrected_the_separator', ('train', 'dev')),
                      ('social_iqa', ('train', 'dev')),
                      ('race_string', ('train', 'dev', 'test'))]
DOMAINS = TRAIN_DOMAINS + TEST_DOMAINS

EXT_DOMAINS = [('squad1_1', ('train', 'dev')),
               ('squad2', ('train', 'dev')),
               ('newsqa', ('train', 'dev')),
               ('quoref', ('train', 'dev')),
               ('ropes', ('train', 'dev'))]
EXT_TRAIN_DOMAINS = [('squad1_1', ('train', 'dev')),
                     ('newsqa', ('train', 'dev'))]
EXT_TEST_DOMAINS = [('squad2', ('train', 'dev')),
                    ('quoref', ('train', 'dev')),
                    ('ropes', ('train', 'dev'))]
MULTI_CHOICE = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)']


def compose_qa_pair(question: str,
                    answer: str,
                    is_neg: bool,
                    neg_method: str='weight') -> Tuple[str, str, Union[float, str]]:
  if neg_method == 'weight':
    if is_neg:
      return question, answer, -1.0
    return question, answer, 1.0
  elif neg_method == 'indicator':
    return question + ' ' + ('False:' if is_neg else 'True:'), answer, 1.0
  elif neg_method == 'bool':
    if is_neg:
      return question, answer, 'False'
    return question, answer, 'True'
  else:
    raise NotImplementedError


def one2multi(in_fname: str, out_fname: str, oneline: bool=False, num_sep: int=10):
  with tf.io.gfile.GFile(in_fname, 'r') as fin, tf.io.gfile.GFile(out_fname, 'w') as fout:
    for line_ind, line in enumerate(fin):
      question, answer = line.strip().split('\t')
      question = question.strip()
      answer = answer.strip()

      cc = 0
      remain_answer = question.split(MULTI_CHOICE[0], 1)  # TODO: assume ABCD is not in question-answer pairs.
      remain_answer = None if len(remain_answer) < 2 else remain_answer[1]
      aipairs = []
      for i in range(len(MULTI_CHOICE)):
        if remain_answer is None:
          break
        if i < len(MULTI_CHOICE) - 1:
          _answer = remain_answer.split(MULTI_CHOICE[i + 1], 1)
        else:
          _answer = [remain_answer]  # remove the context
        remain_answer = None
        if len(_answer) == 2:
          _answer, remain_answer = _answer
        else:
          _answer = _answer[0].split('\\n', 1)[0]
        _answer = _answer.strip()
        q, a, b = compose_qa_pair(question, _answer, is_neg=_answer != answer, neg_method='bool')
        cc += int(b == 'True')
        if b == 'True':
          aipairs.insert(0, (a, i))
        else:
          aipairs.append((a, i))
        if not oneline:
          fout.write('{}\t{}\t{}\t{}\n'.format(line_ind, q, a, b))
      assert cc >= 1, '#correct answer is {}, should >= 1, question is "{}", answer is "{}"'.format(cc, question, answer)
      if oneline:
        fout.write('{}\t{}\t{}\n'.format(
          line_ind,
          question,
          '\t'.join('{}\t{}'.format(a, c) for a, c in (aipairs + [('', '')] * (num_sep - len(aipairs))))))


def multi2one(in_fname: str, out_fname: str, num_sep: int=10):
  with tf.io.gfile.GFile(in_fname, 'r') as fin, tf.io.gfile.GFile(out_fname, 'w') as fout:
    ans_cor_li: List[Tuple[str, str]] = []
    prev_ques = None
    prev_lid = None
    ans_id = 0
    for line in fin:
      lid, ques, ans, cor = line.strip().split('\t')
      if prev_lid != None and lid != prev_lid:
        fout.write('{}\t{}\t{}\n'.format(
          prev_lid, prev_ques,
          '\t'.join('{}\t{}'.format(a, c) for a, c in (ans_cor_li + [('', '')] * (num_sep - len(ans_cor_li))))))
        ans_cor_li = []
        ans_id = 0
      ans_cor_li.append((ans, ans_id))
      prev_ques = ques
      prev_lid = lid
      ans_id += 1
    if len(ans_cor_li) > 0:
      fout.write('{}\t{}\t{}\n'.format(
        lid, ques,
        '\t'.join('{}\t{}'.format(a, c) for a, c in (ans_cor_li + [('', '')] * (num_sep - len(ans_cor_li))))))


@gin.configurable
def build_uq(neg_method: str='indicator', ret_ind: int=0, ret_method: str='q-prepend'):
  for domain, splits in TOY_DOMAINS:
    t5.data.TaskRegistry.add(
      'uq_toy_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_TOY_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_toy_bal_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_TOY_BAL_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
  for domain, splits in HP_MH_DOMAINS:
    t5.data.TaskRegistry.add(
      'uq_mh_mh_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
  for domain, splits in MH_DOMAINS:
    t5.data.TaskRegistry.add(
      'uq_mh_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_first_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_FIRST_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_first_hint_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_FIRST_HINT_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_second_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_SECOND_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_second_path_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_SECOND_PATH_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_second_path_inverse_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_SECOND_PATH_INVERSE_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_second_alltheway_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_SECOND_ALLTHEWAY_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_second_hint_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_SECOND_HINT_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_dev_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_DEV_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_s_m_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_S_M_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh__sm_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH__SM_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_mh_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])

    t5.data.TaskRegistry.add(
      'uq_mh_mh_sql_simple_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_SQL_SIMPLE_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_mh_sql_simple_combine_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_SQL_SIMPLE_COMBINE_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_mh_sql_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_SQL_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_mh_sql_combine_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_SQL_COMBINE_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_mh_sql_combine_nomh_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_SQL_COMBINE_NOMH_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_mh_sql_combine_nonlmh_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_SQL_COMBINE_NONLMH_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_mh_sql_combine_concat_as_mh_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_SQL_COMBINE_CONCAT_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])

    t5.data.TaskRegistry.add(
      'uq_mh_mh_cwq_elq_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_CWQ_ELQ_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_mh_dedup_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_DEDUP_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_mh_path_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_PATH_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_mh_path_inverse_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_PATH_INVERSE_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_mh_hint_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_HINT_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_mh_reducehop_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_REDUCEHOP_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_mh_implicit_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_IMPLICIT_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_mh_implicit_nomh_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_IMPLICIT_NOMH_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_mh_cwq_elq_implicit_nomh_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_CWQ_ELQ_IMPLICIT_NOMH_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_mh_explicit_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_EXPLICIT_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_mh_reducehop_first_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_REDUCEHOP_FIRST_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_mh_reducehop_second_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_REDUCEHOP_SECOND_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_mh_mh_statement_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_MH_MULTIHOP_STATEMENT_GS_OL, domain=domain, num_sep=1),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=1)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])

  t5.data.MixtureRegistry.remove('uq_mh_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_ol_mix', ['uq_mh_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_first_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_first_ol_mix', ['uq_mh_first_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_first_hint_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_first_hint_ol_mix', ['uq_mh_first_hint_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_second_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_second_ol_mix', ['uq_mh_second_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_second_path_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_second_path_ol_mix', ['uq_mh_second_path_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_second_path_inverse_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_second_path_inverse_ol_mix', ['uq_mh_second_path_inverse_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_second_alltheway_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_second_alltheway_ol_mix', ['uq_mh_second_alltheway_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_second_hint_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_second_hint_ol_mix', ['uq_mh_second_hint_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_dev_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_dev_ol_mix', ['uq_mh_dev_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_s_m_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_s_m_ol_mix', ['uq_mh_s_m_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh__sm_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh__sm_ol_mix', ['uq_mh__sm_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_mh_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_ol_mix', ['uq_mh_mh_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)

  t5.data.MixtureRegistry.remove('uq_mh_mh_sql_simple_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_sql_simple_ol_mix', ['uq_mh_mh_sql_simple_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_mh_sql_simple_combine_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_sql_simple_combine_ol_mix', ['uq_mh_mh_sql_simple_combine_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_mh_sql_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_sql_ol_mix', ['uq_mh_mh_sql_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_mh_sql_combine_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_sql_combine_ol_mix', ['uq_mh_mh_sql_combine_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)

  t5.data.MixtureRegistry.remove('uq_mh_mh_sql_combine_nomh_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_sql_combine_nomh_ol_mix', ['uq_mh_mh_sql_combine_nomh_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_mh_sql_combine_nonlmh_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_sql_combine_nonlmh_ol_mix', ['uq_mh_mh_sql_combine_nonlmh_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_mh_sql_combine_concat_as_mh_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_sql_combine_concat_as_mh_ol_mix', ['uq_mh_mh_sql_combine_concat_as_mh_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)

  t5.data.MixtureRegistry.remove('uq_mh_mh_cwq_elq_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_cwq_elq_ol_mix', ['uq_mh_mh_cwq_elq_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_mh_dedup_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_dedup_ol_mix', ['uq_mh_mh_dedup_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_mh_path_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_path_ol_mix', ['uq_mh_mh_path_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_mh_path_inverse_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_path_inverse_ol_mix', ['uq_mh_mh_path_inverse_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_mh_hint_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_hint_ol_mix', ['uq_mh_mh_hint_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_mh_reducehop_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_reducehop_ol_mix', ['uq_mh_mh_reducehop_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_mh_implicit_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_implicit_ol_mix', ['uq_mh_mh_implicit_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_mh_implicit_nomh_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_implicit_nomh_ol_mix', ['uq_mh_mh_implicit_nomh_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_mh_cwq_elq_implicit_nomh_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_cwq_elq_implicit_nomh_ol_mix', ['uq_mh_mh_cwq_elq_implicit_nomh_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_mh_explicit_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_explicit_ol_mix', ['uq_mh_mh_explicit_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_mh_reducehop_first_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_reducehop_first_ol_mix', ['uq_mh_mh_reducehop_first_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_mh_reducehop_second_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_reducehop_second_ol_mix', ['uq_mh_mh_reducehop_second_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_mh_mh_statement_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_statement_ol_mix', ['uq_mh_mh_statement_{}_ol'.format(domain) for domain, _ in MH_DOMAINS], default_rate=1.0)

  t5.data.MixtureRegistry.remove('uq_mh_mh_hp_ol_mix')
  t5.data.MixtureRegistry.add('uq_mh_mh_hp_ol_mix', ['uq_mh_mh_{}_ol'.format(domain) for domain, _ in HP_MH_DOMAINS], default_rate=1.0)

  t5.data.MixtureRegistry.remove('uq_toy_ol_mix')
  t5.data.MixtureRegistry.add('uq_toy_ol_mix', ['uq_toy_{}_ol'.format(domain) for domain, _ in TOY_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_toy_bal_ol_mix')
  t5.data.MixtureRegistry.add('uq_toy_bal_ol_mix', ['uq_toy_bal_{}_ol'.format(domain) for domain, _ in TOY_DOMAINS], default_rate=1.0)

  for domain, splits in DOMAINS:
    # multi-line tasks
    t5.data.TaskRegistry.add(
      'uq_{}'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_PREP_GS, domain=domain, use_neg=True, neg_method=neg_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.MixtureRegistry.remove('uq_{}_mix'.format(domain))
    t5.data.MixtureRegistry.add('uq_{}_mix'.format(domain), ['uq_{}'.format(domain)], default_rate=1.0)
    t5.data.TaskRegistry.add(
      'uq_{}_inp'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_onlyinput_fn, bucket=UNIFIEDQA_PREP_GS, domain=domain),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_ret_drqa_3s'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_ret, bucket=UNIFIEDQA_PREP_GS_RET_DRQA_3S, domain=domain, ret_ind=ret_ind, ret_method=ret_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])

    # multi-line bt tasks
    t5.data.TaskRegistry.add(
      'uq_{}_bt'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_PREP_GS_BT, domain=domain, use_neg=True, neg_method=neg_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_bt_replace'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_PREP_GS_BT_REP, domain=domain, use_neg=True, neg_method=neg_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_bt_dedup_replace'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_PREP_GS_BT_DEDUP_REP, domain=domain, use_neg=True, neg_method=neg_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.MixtureRegistry.remove('uq_{}_bt_dedup_replace_mix'.format(domain))
    t5.data.MixtureRegistry.add('uq_{}_bt_dedup_replace_mix'.format(domain), ['uq_{}_bt_dedup_replace'.format(domain)], default_rate=1.0)
    t5.data.TaskRegistry.add(
      'uq_{}_bt_dedup_top10_replace'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_PREP_GS_BT_DEDUP_TOP10_REP, domain=domain, use_neg=True, neg_method=neg_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_bt_dedup_top20_replace'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_PREP_GS_BT_DEDUP_TOP20_REP, domain=domain, use_neg=True, neg_method=neg_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_ret_drqa_3s_bt_replace'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_ret, bucket=UNIFIEDQA_PREP_GS_RET_DRQA_3S_BT_REP, domain=domain, ret_ind=ret_ind, ret_method=ret_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_ret_drqa_3s_bt_dedup_replace'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_ret, bucket=UNIFIEDQA_PREP_GS_RET_DRQA_3S_BT_DEDUP_REP, domain=domain, ret_ind=ret_ind,
        ret_method=ret_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_ret_drqa_3s_bt_replace_inp'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_ret, bucket=UNIFIEDQA_PREP_GS_RET_DRQA_3S_BT_REP, domain=domain, ret_ind=ret_ind, ret_method=ret_method, onlyinput=True),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_ret_drqa_3s_bt_dedup_replace_inp'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_ret, bucket=UNIFIEDQA_PREP_GS_RET_DRQA_3S_BT_DEDUP_REP, domain=domain, ret_ind=ret_ind,
        ret_method=ret_method, onlyinput=True),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])

    # single-line tasks
    t5.data.TaskRegistry.add(
      'uq_{}_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_PREP_GS_OL, domain=domain, num_sep=len(MULTI_CHOICE)),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=len(MULTI_CHOICE))],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])

  t5.data.TaskRegistry.add(
    'dup_test',
    dataset_fn=functools.partial(
      qa_dataset_fn, bucket=UNIFIEDQA_PREP_GS, domain='dup_test', use_neg=True, neg_method=neg_method),
    splits=('train', 'dev'),
    text_preprocessor=[trivia_preprocessor],
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy])
  t5.data.MixtureRegistry.remove('dup_test_mix')
  t5.data.MixtureRegistry.add('dup_test_mix', ['dup_test'], default_rate=1.0)

  t5.data.MixtureRegistry.remove('uq_train_mix')
  t5.data.MixtureRegistry.add('uq_train_mix', ['uq_{}'.format(domain) for domain, _ in TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_train_inp_mix')
  t5.data.MixtureRegistry.add('uq_train_inp_mix', ['uq_{}_inp'.format(domain) for domain, _ in TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_train_ol_mix')
  t5.data.MixtureRegistry.add('uq_train_ol_mix', ['uq_{}_ol'.format(domain) for domain, _ in TRAIN_DOMAINS], default_rate=1.0)

  t5.data.MixtureRegistry.remove('uq_test_mix')
  t5.data.MixtureRegistry.add('uq_test_mix', ['uq_{}'.format(domain) for domain, _ in TEST_DOMAINS], default_rate=1.0)

  t5.data.MixtureRegistry.remove('uq_sub_test_mix')
  t5.data.MixtureRegistry.add('uq_sub_test_mix', ['uq_{}'.format(domain) for domain, _ in SUB_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_sub_test_inp_mix')
  t5.data.MixtureRegistry.add('uq_sub_test_inp_mix', ['uq_{}_inp'.format(domain) for domain, _ in SUB_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_sub_test_bt_mix')
  t5.data.MixtureRegistry.add('uq_sub_test_bt_mix', ['uq_{}_bt'.format(domain) for domain, _ in SUB_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_sub_test_bt_replace_mix')
  t5.data.MixtureRegistry.add('uq_sub_test_bt_replace_mix', ['uq_{}_bt_replace'.format(domain) for domain, _ in SUB_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_sub_test_ret_drqa_3s_mix')
  t5.data.MixtureRegistry.add('uq_sub_test_ret_drqa_3s_mix', ['uq_{}_ret_drqa_3s'.format(domain) for domain, _ in SUB_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_sub_test_ret_drqa_3s_bt_replace_mix')
  t5.data.MixtureRegistry.add('uq_sub_test_ret_drqa_3s_bt_replace_mix', ['uq_{}_ret_drqa_3s_bt_replace'.format(domain) for domain, _ in SUB_TEST_DOMAINS], default_rate=1.0)

  t5.data.MixtureRegistry.remove('uq_clean_train_ol_mix')
  t5.data.MixtureRegistry.add('uq_clean_train_ol_mix', ['uq_{}_ol'.format(domain) for domain, _ in CLEAN_TRAIN_DOMAINS], default_rate=1.0)

  for split, domains in [('train', CLEAN_TRAIN_DOMAINS), ('test', CLEAN_TEST_DOMAINS)]:
    t5.data.MixtureRegistry.remove('uq_clean_{}_mix'.format(split))
    t5.data.MixtureRegistry.add('uq_clean_{}_mix'.format(split), ['uq_{}'.format(domain) for domain, _ in domains], default_rate=1.0)
    t5.data.MixtureRegistry.remove('uq_clean_{}_inp_mix'.format(split))
    t5.data.MixtureRegistry.add('uq_clean_{}_inp_mix'.format(split), ['uq_{}_inp'.format(domain) for domain, _ in domains], default_rate=1.0)
    t5.data.MixtureRegistry.remove('uq_clean_{}_bt_mix'.format(split))
    t5.data.MixtureRegistry.add('uq_clean_{}_bt_mix'.format(split), ['uq_{}_bt'.format(domain) for domain, _ in domains], default_rate=1.0)
    t5.data.MixtureRegistry.remove('uq_clean_{}_bt_replace_mix'.format(split))
    t5.data.MixtureRegistry.add('uq_clean_{}_bt_replace_mix'.format(split), ['uq_{}_bt_replace'.format(domain) for domain, _ in domains], default_rate=1.0)
    t5.data.MixtureRegistry.remove('uq_clean_{}_bt_dedup_replace_mix'.format(split))
    t5.data.MixtureRegistry.add('uq_clean_{}_bt_dedup_replace_mix'.format(split), ['uq_{}_bt_dedup_replace'.format(domain) for domain, _ in domains], default_rate=1.0)
    t5.data.MixtureRegistry.remove('uq_clean_{}_bt_dedup_top10_replace_mix'.format(split))
    t5.data.MixtureRegistry.add('uq_clean_{}_bt_dedup_top10_replace_mix'.format(split), ['uq_{}_bt_dedup_top10_replace'.format(domain) for domain, _ in domains], default_rate=1.0)
    t5.data.MixtureRegistry.remove('uq_clean_{}_bt_dedup_top20_replace_mix'.format(split))
    t5.data.MixtureRegistry.add('uq_clean_{}_bt_dedup_top20_replace_mix'.format(split), ['uq_{}_bt_dedup_top20_replace'.format(domain) for domain, _ in domains], default_rate=1.0)
    t5.data.MixtureRegistry.remove('uq_clean_{}_ret_drqa_3s_mix'.format(split))
    t5.data.MixtureRegistry.add('uq_clean_{}_ret_drqa_3s_mix'.format(split), ['uq_{}_ret_drqa_3s'.format(domain) for domain, _ in domains],  default_rate=1.0)
    t5.data.MixtureRegistry.remove('uq_clean_{}_ret_drqa_3s_bt_replace_mix'.format(split))
    t5.data.MixtureRegistry.add('uq_clean_{}_ret_drqa_3s_bt_replace_mix'.format(split), ['uq_{}_ret_drqa_3s_bt_replace'.format(domain) for domain, _ in domains], default_rate=1.0)
    t5.data.MixtureRegistry.remove('uq_clean_{}_ret_drqa_3s_bt_dedup_replace_mix'.format(split))
    t5.data.MixtureRegistry.add('uq_clean_{}_ret_drqa_3s_bt_dedup_replace_mix'.format(split), ['uq_{}_ret_drqa_3s_bt_dedup_replace'.format(domain) for domain, _ in domains], default_rate=1.0)
    t5.data.MixtureRegistry.remove('uq_clean_{}_ret_drqa_3s_bt_replace_inp_mix'.format(split))
    t5.data.MixtureRegistry.add('uq_clean_{}_ret_drqa_3s_bt_replace_inp_mix'.format(split), ['uq_{}_ret_drqa_3s_bt_replace_inp'.format(domain) for domain, _ in domains], default_rate=1.0)
    t5.data.MixtureRegistry.remove('uq_clean_{}_ret_drqa_3s_bt_dedup_replace_inp_mix'.format(split))
    t5.data.MixtureRegistry.add('uq_clean_{}_ret_drqa_3s_bt_dedup_replace_inp_mix'.format(split), ['uq_{}_ret_drqa_3s_bt_dedup_replace_inp'.format(domain) for domain, _ in domains], default_rate=1.0)

  t5.data.MixtureRegistry.remove('uq_all_mix')
  t5.data.MixtureRegistry.add('uq_all_mix', ['uq_{}'.format(domain) for domain, _ in TRAIN_DOMAINS + TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_sub_all_mix')
  t5.data.MixtureRegistry.add('uq_sub_all_mix', ['uq_{}'.format(domain) for domain, _ in TRAIN_DOMAINS + SUB_TEST_DOMAINS], default_rate=1.0)

  for domain, splits in EXT_DOMAINS:
    # single-line extractive tasks
    t5.data.TaskRegistry.add(
      'uq_{}_oc'.format(domain),
      dataset_fn=functools.partial(qa_dataset_fn_onlycorrect, bucket=UNIFIEDQA_RAW_GS, domain=domain),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.MixtureRegistry.remove('uq_{}_oc_mix'.format(domain))
    t5.data.MixtureRegistry.add('uq_{}_oc_mix'.format(domain), ['uq_{}_oc'.format(domain)], default_rate=1.0)
    t5.data.TaskRegistry.add(
      'uq_{}_oc_dup'.format(domain),
      dataset_fn=functools.partial(qa_dataset_fn_onlycorrect, bucket=UNIFIEDQA_RAW_DUP_GS, domain=domain),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])

    t5.data.TaskRegistry.add(
      'uq_{}_first_decode_uq3B'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_RAW_FIRST_DECODE_UQ3B_GS, domain=domain, use_neg=True, neg_method=neg_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.MixtureRegistry.remove('uq_{}_first_decode_uq3B_mix'.format(domain))
    t5.data.MixtureRegistry.add('uq_{}_first_decode_uq3B_mix'.format(domain), ['uq_{}_first_decode_uq3B'.format(domain)], default_rate=1.0)

    # multi-line tasks
    t5.data.TaskRegistry.add(
      'uq_{}_decode'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_RAW_DECODE_GS, domain=domain, use_neg=True, neg_method=neg_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_RAW_DECODE_GS + '_uq', domain=domain, use_neg=True, neg_method=neg_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_GS, domain=domain, use_neg=True, neg_method=neg_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_dedup'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS, domain=domain, use_neg=True, neg_method=neg_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_sample'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS, domain=domain, use_neg=True, neg_method=neg_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_span_topk_nogold'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS, domain=domain, use_neg=True,
        neg_method=neg_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_span_topk'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_GS, domain=domain, use_neg=True,
        neg_method=neg_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_inp'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_onlyinput_fn, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_GS, domain=domain),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_dedup_inp'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_onlyinput_fn, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS, domain=domain),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_sample_inp'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_onlyinput_fn, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS, domain=domain),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_span_topk_nogold_inp'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_onlyinput_fn, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS, domain=domain),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_span_topk_inp'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_onlyinput_fn, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_GS, domain=domain),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_ret_drqa_3s'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_ret, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_GS_RET_DRQA_3S, domain=domain, ret_ind=ret_ind, ret_method=ret_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_dedup_ret_drqa_3s'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_ret, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_RET_DRQA_3S, domain=domain, ret_ind=ret_ind,
        ret_method=ret_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_sample_ret_drqa_3s'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_ret, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_RET_DRQA_3S, domain=domain, ret_ind=ret_ind,
        ret_method=ret_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_span_topk_nogold_ret_drqa_3s'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_ret, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS_RET_DRQA_3S, domain=domain, ret_ind=ret_ind,
        ret_method=ret_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_bt'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_GS_BT, domain=domain, use_neg=True, neg_method=neg_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_dedup_bt'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_BT, domain=domain, use_neg=True, neg_method=neg_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_sample_bt'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_BT, domain=domain, use_neg=True,
        neg_method=neg_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_span_topk_nogold_bt'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS_BT, domain=domain, use_neg=True,
        neg_method=neg_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])

    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_ret_drqa_3s_bt'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_ret, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_GS_RET_DRQA_3S_BT, domain=domain, ret_ind=ret_ind,
        ret_method=ret_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_dedup_ret_drqa_3s_bt'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_ret, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_RET_DRQA_3S_BT, domain=domain, ret_ind=ret_ind,
        ret_method=ret_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_sample_ret_drqa_3s_bt'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_ret, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_RET_DRQA_3S_BT, domain=domain, ret_ind=ret_ind,
        ret_method=ret_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_span_topk_nogold_ret_drqa_3s_bt'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_ret, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS_RET_DRQA_3S_BT, domain=domain, ret_ind=ret_ind,
        ret_method=ret_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_ret_drqa_3s_bt_inp'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_ret, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_GS_RET_DRQA_3S_BT, domain=domain, ret_ind=ret_ind,
        ret_method=ret_method, onlyinput=True),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_dedup_ret_drqa_3s_bt_inp'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_ret, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_RET_DRQA_3S_BT, domain=domain, ret_ind=ret_ind,
        ret_method=ret_method, onlyinput=True),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_sample_ret_drqa_3s_bt_inp'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_ret, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_RET_DRQA_3S_BT, domain=domain, ret_ind=ret_ind,
        ret_method=ret_method, onlyinput=True),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_span_topk_nogold_ret_drqa_3s_bt_inp'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_ret, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_NOGOLD_GS_RET_DRQA_3S_BT, domain=domain, ret_ind=ret_ind,
        ret_method=ret_method, onlyinput=True),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])

    # single-line tasks
    t5.data.TaskRegistry.add(
      'uq_{}_decode_ol'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_RAW_DECODE_GS_OL, domain=domain, num_sep=5),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=5)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_ol_uq3B'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_GS_OL, domain=domain, num_sep=5),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=5)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_ol_uq3B_dedup'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_OL, domain=domain, num_sep=5),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=5)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_ol_uq3B_sample'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_SAMPLE_GS_OL, domain=domain, num_sep=5),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=5)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_ol_uq3B_span_topk'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_SPAN_TOPK_GS_OL, domain=domain, num_sep=5),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=5)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])

  t5.data.MixtureRegistry.remove('uq_ext_mix')
  t5.data.MixtureRegistry.add('uq_ext_mix', ['uq_{}_oc'.format(domain) for domain, _ in EXT_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_dup_mix')
  t5.data.MixtureRegistry.add('uq_ext_dup_mix', ['uq_{}_oc_dup'.format(domain) for domain, _ in EXT_DOMAINS], default_rate=1.0)

  t5.data.MixtureRegistry.remove('uq_ext_decode_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_mix', ['uq_{}_decode'.format(domain) for domain, _ in EXT_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_uq3B_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_uq3B_mix', ['uq_{}_decode_uq3B'.format(domain) for domain, _ in EXT_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_uq_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_uq_mix', ['uq_{}_decode_uq'.format(domain) for domain, _ in EXT_DOMAINS], default_rate=1.0)

  t5.data.MixtureRegistry.remove('uq_ext_decode_train_ol_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_ol_mix', ['uq_{}_decode_ol'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_ol_uq3B_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_ol_uq3B_mix', ['uq_{}_decode_ol_uq3B'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_ol_uq3B_dedup_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_ol_uq3B_dedup_mix', ['uq_{}_decode_ol_uq3B_dedup'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_ol_uq3B_sample_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_ol_uq3B_sample_mix', ['uq_{}_decode_ol_uq3B_sample'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_ol_uq3B_span_topk_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_ol_uq3B_span_topk_mix', ['uq_{}_decode_ol_uq3B_span_topk'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)

  t5.data.MixtureRegistry.remove('uq_ext_first_decode_uq3B_mix')
  t5.data.MixtureRegistry.add('uq_ext_first_decode_uq3B_mix', ['uq_{}_first_decode_uq3B'.format(domain) for domain, _ in EXT_DOMAINS], default_rate=1.0)

  for split, domains in [('train', EXT_TRAIN_DOMAINS), ('test', EXT_TEST_DOMAINS)]:
    for version in ['uq3B', 'uq3B_dedup', 'uq3B_sample', 'uq3B_span_topk_nogold']:
      t5.data.MixtureRegistry.remove('uq_ext_decode_{}_{}_mix'.format(split, version))
      t5.data.MixtureRegistry.add('uq_ext_decode_{}_{}_mix'.format(split, version), ['uq_{}_decode_{}'.format(domain, version) for domain, _ in domains], default_rate=1.0)
      t5.data.MixtureRegistry.remove('uq_ext_decode_{}_{}_inp_mix'.format(split, version))
      t5.data.MixtureRegistry.add('uq_ext_decode_{}_{}_inp_mix'.format(split, version), ['uq_{}_decode_{}_inp'.format(domain, version) for domain, _ in domains], default_rate=1.0)
      t5.data.MixtureRegistry.remove('uq_ext_decode_{}_{}_ret_drqa_3s_mix'.format(split, version))
      t5.data.MixtureRegistry.add('uq_ext_decode_{}_{}_ret_drqa_3s_mix'.format(split, version), ['uq_{}_decode_{}_ret_drqa_3s'.format(domain, version) for domain, _ in domains], default_rate=1.0)
      t5.data.MixtureRegistry.remove('uq_ext_decode_{}_{}_bt_mix'.format(split, version))
      t5.data.MixtureRegistry.add('uq_ext_decode_{}_{}_bt_mix'.format(split, version), ['uq_{}_decode_{}_bt'.format(domain, version) for domain, _ in domains], default_rate=1.0)
      t5.data.MixtureRegistry.remove('uq_ext_decode_{}_{}_ret_drqa_3s_bt_mix'.format(split, version))
      t5.data.MixtureRegistry.add('uq_ext_decode_{}_{}_ret_drqa_3s_bt_mix'.format(split, version), ['uq_{}_decode_{}_ret_drqa_3s_bt'.format(domain, version) for domain, _ in domains], default_rate=1.0)
      t5.data.MixtureRegistry.remove('uq_ext_decode_{}_{}_ret_drqa_3s_bt_inp_mix'.format(split, version))
      t5.data.MixtureRegistry.add('uq_ext_decode_{}_{}_ret_drqa_3s_bt_inp_mix'.format(split, version), ['uq_{}_decode_{}_ret_drqa_3s_bt_inp'.format(domain, version) for domain, _ in domains], default_rate=1.0)

  t5.data.MixtureRegistry.remove('uq_ext_decode_train_uq3B_span_topk_mix'.format(split, version))
  t5.data.MixtureRegistry.add('uq_ext_decode_train_uq3B_span_topk_mix'.format(split, version), ['uq_{}_decode_uq3B_span_topk'.format(domain, version) for domain, _ in domains], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_uq3B_span_topk_inp_mix'.format(split, version))
  t5.data.MixtureRegistry.add('uq_ext_decode_train_uq3B_span_topk_inp_mix'.format(split, version), ['uq_{}_decode_uq3B_span_topk_inp'.format(domain, version) for domain, _ in domains], default_rate=1.0)
