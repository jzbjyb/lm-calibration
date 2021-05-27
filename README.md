# LM Calibration 

This repository contains code for the paper [How Can We Know When Language Models Know? On the Calibration of Language Models for Question Answering](https://arxiv.org/abs/2012.00955)

## Install

Our code is mainly based on [T5][t5] and [mesh-tensorflow][mesh] and runs on TPUs.
Please follow the original [T5][t5] repository to properly setup TPUs.
To install required packages, download [T5][t5] (version 0.6.4) and [mesh-tensorflow][mesh] (version 0.1.16) and copy source files into the `t5` and `mesh_tensorflow` folder.
Don't replace files already in these folders because those files are the files we modified for calibration purpose.

## Fine-tune

Run the following commands to fine-tune the [UnifiedQA][uq] models with `softmax` or `margin` objective functions.
`$tpu` specifies the name of the TPU, `$model_output` specifies the output location to save the fine-tuned model, `$objective` specifies the objective function to use.
```shell
./finetune.sh $tpu 3B $model_output $objective uq_clean_train_ol_mix train mc
```

## Evaluate candidate answers

Run the following commands to evaluate the probabilities of candidate answers.
`$score_output` specifies the location to save the output, and `1103000` specifies the checkpoint to use.
```shell
./score.sh $tpu $score_output $model_output 1103000 uq_clean_test dev
```

## Compute ECE

Run the following commands to compute the ECE metric given the probabilities of candidate answers.
```shell
python cal.py --mix uq_clean_test --split dev --score $score_output
```

[t5]: https://github.com/google-research/text-to-text-transfer-transformer
[mesh]: https://github.com/tensorflow/mesh
[uq]: https://github.com/allenai/unifiedqa
