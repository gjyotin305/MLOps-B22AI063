# Set B: ResNet-18 Evaluation Results (CIFAR-10)

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

## Final Metrics

| Metric | Value |
| --- | --- |
| Final Eval Loss | **0.6689** |
| Final Eval Accuracy | **79.10%** |
| Final Eval Macro F1 | **0.7954** |

## Confusion Matrix

```text
[[75  1 14  3  1  0  0  0  6  0]
 [ 0 92  3  3  0  0  0  0  1  1]
 [ 2  0 85  4  1  5  3  0  0  0]
 [ 0  0  6 67  3 24  0  0  0  0]
 [ 1  0 10  3 80  5  1  0  0  0]
 [ 0  0  1  4  1 93  1  0  0  0]
 [ 0  0 16  7  1  4 72  0  0  0]
 [ 1  0  9  1 10 18  0 61  0  0]
 [ 3  0  4  0  0  1  0  0 92  0]
 [ 3  9  2  3  0  5  1  0  3 74]]
```

## Class-wise Accuracy

| Class | Accuracy |
| --- | --- |
| airplane | 75.00% |
| automobile | 92.00% |
| bird | 85.00% |
| cat | 67.00% |
| deer | 80.00% |
| dog | 93.00% |
| frog | 72.00% |
| horse | 61.00% |
| ship | 92.00% |
| truck | 74.00% |

## Classification Report

```text
              precision    recall  f1-score   support

    airplane       0.88      0.75      0.81       100
  automobile       0.90      0.92      0.91       100
        bird       0.57      0.85      0.68       100
         cat       0.71      0.67      0.69       100
        deer       0.82      0.80      0.81       100
         dog       0.60      0.93      0.73       100
        frog       0.92      0.72      0.81       100
       horse       1.00      0.61      0.76       100
        ship       0.90      0.92      0.91       100
       truck       0.99      0.74      0.85       100

    accuracy                           0.79      1000
   macro avg       0.83      0.79      0.80      1000
weighted avg       0.83      0.79      0.80      1000
```
