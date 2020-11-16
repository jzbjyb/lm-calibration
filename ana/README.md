HTML visualization of success and failure cases.

## The format of each question-answer example is as follow:

```
● The name of the dataset
Question
Candidate answers
Context (if any)
Retrieved sentences (starts with "retrieved:" if any)
→ Score of each candidate answer (The one starting with "♦" is the one we succeed/fail. Paraphrases are separated by spaces and probabilities are at the end.)
```
For successful cases, the score of each candidate answer has three parts: the raw LM probability before applying this method, the raw LM probability after applying this method, and the gap between them (after normalization).

For failure cases, the score of each candidate answer is the probability.

The best model is using margin-based fine-tuning + paraphrasing + retrieval + temparature-based scaling.

## Files

- [Examples improved by paraphrasing](http://jzb.vanpersie.cc/exp/calibration/bt/logprobs-improve.html)
- [Examples improved by retrieval](http://jzb.vanpersie.cc/exp/calibration/ret/logprobs-improve.html)
- [Examples improved by margin-based fine-tuning](http://jzb.vanpersie.cc/exp/calibration/margin/logprobs-improve.html)
- [Examples improved by temparature-based scaling](http://jzb.vanpersie.cc/exp/calibration/temp/logprobs-improve.html)
- [Examples improved by feature-based decision tree](http://jzb.vanpersie.cc/exp/calibration/xgb/logprobs-improve.html)
- [Examples improved by the best model](http://jzb.vanpersie.cc/exp/calibration/ret_bt_temp/logprobs-improve.html)
- [Failure examples using the best model (under estimation)](http://jzb.vanpersie.cc/exp/calibration/fail/logprobs-under.html)
- [Failure examples using the best model (over estimation)](http://jzb.vanpersie.cc/exp/calibration/fail/logprobs-over.html)
