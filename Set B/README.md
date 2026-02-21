# Set B: ResNet-18 Evaluation Results (CIFAR-10)

## How to build

```bash
docker build -t run_eval_minor:v1_eval -f Dockerfile.eval .
docker build -t run_eval_minor:v1_train -f Dockerfile .
```

## How to run

```bash
docker run run_eval_minor:v1_eval 
docker run run_eval_minor:v1_train
```

### Final Metrics

| Metric | Value |
| --- | --- |
| Overall Accuracy | **96.64%** |
| F1 Score | **0.9658** |

### Confusion Matrix

```text
[[488   0   0   0   0   1   1   0   0   0]
 [  0 565   0   1   0   0   0   2   0   0]
 [  0   1 496  17   0   0   0   1   1   0]
 [  0   0   0 505   0   0   0   0   0   0]
 [  0   0   0   0 478   0   1   2   1   9]
 [  0   1   0  71   0 369   0   0   4   1]
 [  1   2   0   0   0   1 471   0   4   0]
 [  1   1  10  15   1   0   0 485   1   0]
 [  0   0   0   0   0   0   0   0 486   1]
 [  1   0   0   4   4   1   0   1   4 490]]
```

### Classification Report

```text
              precision    recall  f1-score   support

           0       0.99      1.00      0.99       490
           1       0.99      0.99      0.99       568
           2       0.98      0.96      0.97       516
           3       0.82      1.00      0.90       505
           4       0.99      0.97      0.98       491
           5       0.99      0.83      0.90       446
           6       1.00      0.98      0.99       479
           7       0.99      0.94      0.97       514
           8       0.97      1.00      0.98       487
           9       0.98      0.97      0.97       505

    accuracy                           0.97      5001
   macro avg       0.97      0.96      0.97      5001
weighted avg       0.97      0.97      0.97      5001
```

### Single Image Test

```text
Image: data/test/7/6561.png
Predicted Class: 7
Confidence: 99.95%
```
