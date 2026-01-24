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

| Model     | Batch Size | Optimizer |     LR | Epochs | pin_memory | Test Accuracy (%) | Train Time (ms) | FLOPs (GFLOPs) | Params (M) |
| --------- | ---------: | --------- | -----: | -----: | ---------: | ----------------: | --------------: | -------------: | ---------: |
| ResNet-18 |         16 | Adam      |  0.001 |      5 |       True |             92.07 |         407,906 |         0.0332 |      11.18 |
| ResNet-18 |         16 | Adam      |  0.001 |     15 |      False |             92.66 |       1,233,524 |         0.0332 |      11.18 |
| ResNet-18 |         16 | Adam      |  0.001 |     15 |       True |             91.88 |       1,237,956 |         0.0332 |      11.18 |
| ResNet-18 |         16 | Adam      | 0.0001 |      5 |      False |             92.27 |         459,555 |         0.0332 |      11.18 |
| ResNet-18 |         16 | Adam      | 0.0001 |      5 |       True |             92.16 |         446,441 |         0.0332 |      11.18 |
| ResNet-34 |         16 | SGD       |  0.001 |      5 |       True |             91.81 |         735,832 |         0.0699 |      21.28 |
| ResNet-34 |         16 | Adam      |  0.001 |      5 |       True |             89.32 |       1,789,542 |         0.0699 |      21.28 |
| ResNet-34 |         32 | SGD       |  0.001 |      5 |       True |             85.69 |         627,229 |         0.0699 |      21.28 |
| ResNet-34 |         32 | Adam      |  0.001 |      5 |       True |             91.42 |       1,136,055 |         0.0699 |      21.28 |
| ResNet-50 |         16 | SGD       |  0.001 |      5 |       True |             90.73 |       1,899,157 |         0.0788 |      23.52 |

