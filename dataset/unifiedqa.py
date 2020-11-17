from typing import Tuple, Union, List, Set
import functools
import tensorflow as tf
import t5
import gin
from .utils import trivia_preprocessor, qa_dataset_fn, qa_dataset_fn_oneline, \
  qa_dataset_fn_onlycorrect, concat_preprocessor, qa_dataset_fn_ret, qa_dataset_onlyinput_fn


UNIFIEDQA_GS = 'gs://unifiedqa/data'
UNIFIEDQA_PREP_GS = 'gs://neulab-qa/data/unifiedqa'
UNIFIEDQA_PREP_GS_RET_DRQA = 'gs://neulab-qa/data/unifiedqa_ret_drqa'
UNIFIEDQA_PREP_GS_RET_DRQA_3S = 'gs://neulab-qa/data/unifiedqa_ret_drqa_3s'
UNIFIEDQA_RAW_GS = 'gs://neulab-qa/unifiedqa/data'
UNIFIEDQA_PREP_GS_OL = 'gs://neulab-qa/data/unifiedqa_oneline'
UNIFIEDQA_PREP_GS_BT = 'gs://neulab-qa/data/unifiedqa_bt'
UNIFIEDQA_PREP_GS_BT_DEDUP = 'gs://neulab-qa/data/unifiedqa_bt_dedup'
UNIFIEDQA_PREP_GS_BT_REP = 'gs://neulab-qa/data/unifiedqa_bt_replace'
UNIFIEDQA_PREP_GS_BT_DEDUP_REP = 'gs://neulab-qa/data/unifiedqa_bt_dedup_replace'
UNIFIEDQA_PREP_GS_RET_DRQA_3S_BT_REP = 'gs://neulab-qa/data/unifiedqa_ret_drqa_3s_bt_replace'
UNIFIEDQA_PREP_GS_RET_DRQA_3S_BT_DEDUP_REP = 'gs://neulab-qa/data/unifiedqa_ret_drqa_3s_bt_dedup_replace'

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
    t5.data.MixtureRegistry.remove('uq_{}_bt_replace_mix'.format(domain))
    t5.data.MixtureRegistry.add('uq_{}_bt_replace_mix'.format(domain), ['uq_{}_bt_replace'.format(domain)], default_rate=1.0)
    t5.data.TaskRegistry.add(
      'uq_{}_bt_dedup_replace'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_PREP_GS_BT_DEDUP_REP, domain=domain, use_neg=True, neg_method=neg_method),
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

  t5.data.MixtureRegistry.remove('uq_clean_train_mix')
  t5.data.MixtureRegistry.add('uq_clean_train_mix', ['uq_{}'.format(domain) for domain, _ in CLEAN_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_clean_train_inp_mix')
  t5.data.MixtureRegistry.add('uq_clean_train_inp_mix', ['uq_{}_inp'.format(domain) for domain, _ in CLEAN_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_clean_train_ol_mix')
  t5.data.MixtureRegistry.add('uq_clean_train_ol_mix', ['uq_{}_ol'.format(domain) for domain, _ in CLEAN_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_clean_train_bt_mix')
  t5.data.MixtureRegistry.add('uq_clean_train_bt_mix', ['uq_{}_bt'.format(domain) for domain, _ in CLEAN_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_clean_train_bt_replace_mix')
  t5.data.MixtureRegistry.add('uq_clean_train_bt_replace_mix', ['uq_{}_bt_replace'.format(domain) for domain, _ in CLEAN_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_clean_train_bt_dedup_replace_mix')
  t5.data.MixtureRegistry.add('uq_clean_train_bt_dedup_replace_mix', ['uq_{}_bt_dedup_replace'.format(domain) for domain, _ in CLEAN_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_clean_train_ret_drqa_3s_mix')
  t5.data.MixtureRegistry.add('uq_clean_train_ret_drqa_3s_mix', ['uq_{}_ret_drqa_3s'.format(domain) for domain, _ in CLEAN_TRAIN_DOMAINS],  default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_clean_train_ret_drqa_3s_bt_replace_mix')
  t5.data.MixtureRegistry.add('uq_clean_train_ret_drqa_3s_bt_replace_mix', ['uq_{}_ret_drqa_3s_bt_replace'.format(domain) for domain, _ in CLEAN_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_clean_train_ret_drqa_3s_bt_dedup_replace_mix')
  t5.data.MixtureRegistry.add('uq_clean_train_ret_drqa_3s_bt_dedup_replace_mix', ['uq_{}_ret_drqa_3s_bt_dedup_replace'.format(domain) for domain, _ in CLEAN_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_clean_train_ret_drqa_3s_bt_replace_inp_mix')
  t5.data.MixtureRegistry.add('uq_clean_train_ret_drqa_3s_bt_replace_inp_mix', ['uq_{}_ret_drqa_3s_bt_replace_inp'.format(domain) for domain, _ in CLEAN_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_clean_train_ret_drqa_3s_bt_dedup_replace_inp_mix')
  t5.data.MixtureRegistry.add('uq_clean_train_ret_drqa_3s_bt_dedup_replace_inp_mix', ['uq_{}_ret_drqa_3s_bt_dedup_replace_inp'.format(domain) for domain, _ in CLEAN_TRAIN_DOMAINS], default_rate=1.0)

  t5.data.MixtureRegistry.remove('uq_clean_test_mix')
  t5.data.MixtureRegistry.add('uq_clean_test_mix', ['uq_{}'.format(domain) for domain, _ in CLEAN_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_clean_test_inp_mix')
  t5.data.MixtureRegistry.add('uq_clean_test_inp_mix', ['uq_{}_inp'.format(domain) for domain, _ in CLEAN_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_clean_test_bt_mix')
  t5.data.MixtureRegistry.add('uq_clean_test_bt_mix', ['uq_{}_bt'.format(domain) for domain, _ in CLEAN_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_clean_test_bt_replace_mix')
  t5.data.MixtureRegistry.add('uq_clean_test_bt_replace_mix', ['uq_{}_bt_replace'.format(domain) for domain, _ in CLEAN_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_clean_test_bt_dedup_replace_mix')
  t5.data.MixtureRegistry.add('uq_clean_test_bt_dedup_replace_mix', ['uq_{}_bt_dedup_replace'.format(domain) for domain, _ in CLEAN_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_clean_test_ret_drqa_3s_mix')
  t5.data.MixtureRegistry.add('uq_clean_test_ret_drqa_3s_mix', ['uq_{}_ret_drqa_3s'.format(domain) for domain, _ in CLEAN_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_clean_test_ret_drqa_3s_bt_replace_mix')
  t5.data.MixtureRegistry.add('uq_clean_test_ret_drqa_3s_bt_replace_mix', ['uq_{}_ret_drqa_3s_bt_replace'.format(domain) for domain, _ in CLEAN_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_clean_test_ret_drqa_3s_bt_dedup_replace_mix')
  t5.data.MixtureRegistry.add('uq_clean_test_ret_drqa_3s_bt_dedup_replace_mix', ['uq_{}_ret_drqa_3s_bt_dedup_replace'.format(domain) for domain, _ in CLEAN_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_clean_test_ret_drqa_3s_bt_replace_inp_mix')
  t5.data.MixtureRegistry.add('uq_clean_test_ret_drqa_3s_bt_replace_inp_mix', ['uq_{}_ret_drqa_3s_bt_replace_inp'.format(domain) for domain, _ in CLEAN_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_clean_test_ret_drqa_3s_bt_dedup_replace_inp_mix')
  t5.data.MixtureRegistry.add('uq_clean_test_ret_drqa_3s_bt_dedup_replace_inp_mix', ['uq_{}_ret_drqa_3s_bt_dedup_replace_inp'.format(domain) for domain, _ in CLEAN_TEST_DOMAINS], default_rate=1.0)

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
      'uq_{}_decode_ans'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_RAW_DECODE_GS_ANS, domain=domain, use_neg=True, neg_method=neg_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_ans_no'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_RAW_DECODE_GS_ANS_NO, domain=domain, use_neg=True, neg_method=neg_method),
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
      'uq_{}_decode_uq3B_ret_drqa_3s_bt_inp'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_onlyinput_fn, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_GS_RET_DRQA_3S_BT, domain=domain),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq3B_dedup_ret_drqa_3s_bt_inp'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_onlyinput_fn, bucket=UNIFIEDQA_RAW_DECODE_UQ3B_DEDUP_GS_RET_DRQA_3S_BT, domain=domain),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq_ft_softmax'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_RAW_DECODE_GS + '_uq_ft_softmax', domain=domain, use_neg=True, neg_method=neg_method),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_uq_ft_margin'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn, bucket=UNIFIEDQA_RAW_DECODE_GS + '_uq_ft_margin', domain=domain, use_neg=True,
        neg_method=neg_method),
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
      'uq_{}_decode_ol_ans'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_RAW_DECODE_GS_OL_ANS, domain=domain, num_sep=5),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=5)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])
    t5.data.TaskRegistry.add(
      'uq_{}_decode_ol_ans_no'.format(domain),
      dataset_fn=functools.partial(
        qa_dataset_fn_oneline, bucket=UNIFIEDQA_RAW_DECODE_GS_OL_ANS_NO, domain=domain, num_sep=5),
      splits=splits,
      text_preprocessor=[trivia_preprocessor],
      token_preprocessor=[functools.partial(concat_preprocessor, num_sep=5)],
      postprocess_fn=t5.data.postprocessors.lower_text,
      metric_fns=[t5.evaluation.metrics.accuracy])

  t5.data.MixtureRegistry.remove('uq_ext_mix')
  t5.data.MixtureRegistry.add('uq_ext_mix', ['uq_{}_oc'.format(domain) for domain, _ in EXT_DOMAINS], default_rate=1.0)

  t5.data.MixtureRegistry.remove('uq_ext_decode_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_mix', ['uq_{}_decode'.format(domain) for domain, _ in EXT_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_uq3B_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_uq3B_mix', ['uq_{}_decode_uq3B'.format(domain) for domain, _ in EXT_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_uq_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_uq_mix', ['uq_{}_decode_uq'.format(domain) for domain, _ in EXT_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_uq_ft_softmax_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_uq_ft_softmax_mix', ['uq_{}_decode_uq_ft_softmax'.format(domain) for domain, _ in EXT_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_uq_ft_margin_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_uq_ft_margin_mix', ['uq_{}_decode_uq_ft_margin'.format(domain) for domain, _ in EXT_DOMAINS], default_rate=1.0)

  t5.data.MixtureRegistry.remove('uq_ext_decode_train_ol_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_ol_mix', ['uq_{}_decode_ol'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_ol_uq3B_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_ol_uq3B_mix', ['uq_{}_decode_ol_uq3B'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_ol_uq3B_dedup_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_ol_uq3B_dedup_mix', ['uq_{}_decode_ol_uq3B_dedup'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_uq3B_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_uq3B_mix', ['uq_{}_decode_uq3B'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_uq3B_dedup_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_uq3B_dedup_mix', ['uq_{}_decode_uq3B_dedup'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_uq3B_inp_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_uq3B_inp_mix', ['uq_{}_decode_uq3B_inp'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_uq3B_dedup_inp_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_uq3B_dedup_inp_mix', ['uq_{}_decode_uq3B_dedup_inp'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_ol_ans_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_ol_ans_mix', ['uq_{}_decode_ol_ans'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_ol_ans_no_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_ol_ans_no_mix', ['uq_{}_decode_ol_ans_no'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_uq3B_ret_drqa_3s_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_uq3B_ret_drqa_3s_mix', ['uq_{}_decode_uq3B_ret_drqa_3s'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_uq3B_dedup_ret_drqa_3s_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_uq3B_dedup_ret_drqa_3s_mix', ['uq_{}_decode_uq3B_dedup_ret_drqa_3s'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_uq3B_bt_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_uq3B_bt_mix', ['uq_{}_decode_uq3B_bt'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_uq3B_dedup_bt_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_uq3B_dedup_bt_mix', ['uq_{}_decode_uq3B_dedup_bt'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_uq3B_ret_drqa_3s_bt_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_uq3B_ret_drqa_3s_bt_mix', ['uq_{}_decode_uq3B_ret_drqa_3s_bt'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_uq3B_dedup_ret_drqa_3s_bt_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_uq3B_dedup_ret_drqa_3s_bt_mix', ['uq_{}_decode_uq3B_dedup_ret_drqa_3s_bt'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_uq3B_ret_drqa_3s_bt_inp_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_uq3B_ret_drqa_3s_bt_inp_mix', ['uq_{}_decode_uq3B_ret_drqa_3s_bt_inp'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_uq3B_dedup_ret_drqa_3s_bt_inp_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_uq3B_dedup_ret_drqa_3s_bt_inp_mix', ['uq_{}_decode_uq3B_dedup_ret_drqa_3s_bt_inp'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)

  t5.data.MixtureRegistry.remove('uq_ext_decode_test_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_test_mix', ['uq_{}_decode'.format(domain) for domain, _ in EXT_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_test_uq3B_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_test_uq3B_mix', ['uq_{}_decode_uq3B'.format(domain) for domain, _ in EXT_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_test_uq3B_dedup_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_test_uq3B_dedup_mix', ['uq_{}_decode_uq3B_dedup'.format(domain) for domain, _ in EXT_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_test_uq3B_inp_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_test_uq3B_inp_mix', ['uq_{}_decode_uq3B_inp'.format(domain) for domain, _ in EXT_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_test_uq3B_dedup_inp_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_test_uq3B_dedup_inp_mix', ['uq_{}_decode_uq3B_dedup_inp'.format(domain) for domain, _ in EXT_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_test_ans_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_test_ans_mix', ['uq_{}_decode_ans'.format(domain) for domain, _ in EXT_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_test_ans_no_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_test_ans_no_mix', ['uq_{}_decode_ans_no'.format(domain) for domain, _ in EXT_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_test_uq3B_ret_drqa_3s_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_test_uq3B_ret_drqa_3s_mix', ['uq_{}_decode_uq3B_ret_drqa_3s'.format(domain) for domain, _ in EXT_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_test_uq3B_dedup_ret_drqa_3s_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_test_uq3B_dedup_ret_drqa_3s_mix', ['uq_{}_decode_uq3B_dedup_ret_drqa_3s'.format(domain) for domain, _ in EXT_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_test_uq3B_bt_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_test_uq3B_bt_mix', ['uq_{}_decode_uq3B_bt'.format(domain) for domain, _ in EXT_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_test_uq3B_dedup_bt_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_test_uq3B_dedup_bt_mix', ['uq_{}_decode_uq3B_dedup_bt'.format(domain) for domain, _ in EXT_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_test_uq3B_ret_drqa_3s_bt_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_test_uq3B_ret_drqa_3s_bt_mix', ['uq_{}_decode_uq3B_ret_drqa_3s_bt'.format(domain) for domain, _ in EXT_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_test_uq3B_dedup_ret_drqa_3s_bt_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_test_uq3B_dedup_ret_drqa_3s_bt_mix', ['uq_{}_decode_uq3B_dedup_ret_drqa_3s_bt'.format(domain) for domain, _ in EXT_TEST_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_uq3B_ret_drqa_3s_bt_inp_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_uq3B_ret_drqa_3s_bt_inp_mix', ['uq_{}_decode_uq3B_ret_drqa_3s_bt_inp'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
  t5.data.MixtureRegistry.remove('uq_ext_decode_train_uq3B_dedup_ret_drqa_3s_bt_inp_mix')
  t5.data.MixtureRegistry.add('uq_ext_decode_train_uq3B_dedup_ret_drqa_3s_bt_inp_mix', ['uq_{}_decode_uq3B_dedup_ret_drqa_3s_bt_inp'.format(domain) for domain, _ in EXT_TRAIN_DOMAINS], default_rate=1.0)
