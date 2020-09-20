#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple
import functools
import re
import sys
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import t5
from t5.models.mesh_transformer_main import console_entry_point

TEST_GS = 'gs://neulab-qa/data/test'
IND2CHAR = dict(zip(range(4), 'ABCD'))
CHAR2IND = dict(zip('ABCD', range(4)))

def compose_qa_pair(question: str, choices: List[str], answer: str) -> Tuple[str, str]:
    choices = tf.strings.join([tf.strings.join(['(', IND2CHAR[i], ') ', c])
                               for i, c in enumerate(choices)], separator=' ')
    question = tf.strings.join(['The following are multiple choice questions.',
                                question, choices, 'Answer:'], separator=' ')
    return question, answer

def test_dataset_fn(split, shuffle_files=False):
    csv_dir = os.path.join(TEST_GS, split)
    csv_files = tf.io.gfile.listdir(csv_dir)
    csv_files = [os.path.join(csv_dir, cf) for cf in csv_files]

    ds = tf.data.experimental.CsvDataset(csv_files, record_defaults=['', '', '', '', '', ''])

    # Convert each tuple to a {"question": ... "answer": ...} dict.
    ds = ds.map(lambda *ex: dict(zip(['question', 'answer'], compose_qa_pair(ex[0], ex[1:5], ex[5]))))

    return ds

def trivia_preprocessor(ds):
    def normalize_text(text):
        text = tf.strings.lower(text)
        return text

    def to_inputs_and_targets(ex):
        return {
            'inputs': normalize_text(ex['question']),
            'targets': normalize_text(ex['answer'])
        }

    return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)

t5.data.TaskRegistry.add(
    'multi_test',
    dataset_fn=test_dataset_fn,
    splits=['dev', 'val', 'test'],
    text_preprocessor=[trivia_preprocessor],
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy],
    # TODO: add num_input_examples
)

t5.data.MixtureRegistry.remove('multi_test_mix')
t5.data.MixtureRegistry.add( 'multi_test_mix', ['multi_test'], default_rate=1.0)

if __name__ == '__main__':

    print('\n\n\n')
    print('======= A few dev examples ... =======')
    for ex in tfds.as_numpy(test_dataset_fn('dev').take(5)):
        print(ex)
    print('\n\n\n')

    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(console_entry_point())
