# MLOps-B22AI063
MLOps-Assignments 

## Question 1 

### Part A: MNIST and FashionMNIST ABLATIONS

| Batch Size | Optimizer | Learning Rate | Epochs | pin_memory | ResNet-18 Accuracy (%) | ResNet-50 Accuracy (%) |
| ---------: | --------- | ------------: | -----: | ---------: | ---------------------: | ---------------------: |
|         16 | SGD       |         0.001 |     05 |      False |                  99.18% |                  98.98% |
|         16 | SGD       |        0.0001 |     15 |       True |                  99.48% |                  98.85% |
|         16 | Adam      |         0.001 |     05 |       True |                  99.38% |                  98.17% |
|         16 | Adam      |        0.0001 |     15 |      False |                  99.32% |                  99.25% |
|         32 | SGD       |         0.001 |     05 |      False |                  99.28% |                  99.71% |
|         32 | SGD       |        0.0001 |     15 |       True |                  99.19% |                  99.48% |
|         32 | Adam      |         0.001 |     05 |       True |                  99.32% |                  98.97% |
|         32 | Adam      |        0.0001 |     15 |      False |                  99.14% |                  99.06% |



### Part B: SVM 

| Dataset      | Kernel | Train Samples | Test Accuracy (%) | Training Time (ms) |
| ------------ | ------ | ------------: | ----------------: | -----------------: |
| MNIST        | RBF    |        10,000 |            95.94% |            7150.62 |
| MNIST        | Poly   |        10,000 |            95.15% |            7986.79 |
| FashionMNIST | RBF    |        10,000 |            85.31% |            8506.86 |
| FashionMNIST | Poly   |        10,000 |            81.66% |            7345.85 |


## Question 2
