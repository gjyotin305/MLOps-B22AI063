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

The results from Question 1(a) show that both ResNet-18 and ResNet-50 achieve extremely high classification accuracy on MNIST, consistently above 99% across all hyperparameter settings. This indicates that MNIST is a relatively simple dataset for modern convolutional architectures, and performance saturates quickly regardless of optimizer choice or batch size. Small variations can still be observed: SGD with a slightly higher learning rate (0.001) tends to provide strong stability, while Adam achieves comparable accuracy with faster convergence in fewer epochs. The effect of pin_memory is minimal in terms of accuracy, suggesting that it primarily influences data loading efficiency rather than model generalization. Overall, both architectures perform nearly optimally, with ResNet-50 occasionally providing marginal improvements but not a significant advantage given the simplicity of the dataset.

In Question 1(b), the SVM baselines demonstrate strong performance on MNIST, reaching around 96% accuracy with the RBF kernel, but accuracy drops significantly on FashionMNIST, where the best result is 85.31%. This highlights the increased complexity of FashionMNIST, where class boundaries are less separable in raw pixel space and require deeper feature extraction. The RBF kernel consistently outperforms the polynomial kernel, confirming that non-linear decision boundaries are better suited for these datasets. However, compared to deep ResNet models, SVM performance remains limited, especially for FashionMNIST, showing the benefit of representation learning in CNNs.

The results from Question 2 further emphasize the trade-off between model complexity and efficiency on FashionMNIST. ResNet-18 achieves the best accuracy (92.66%) with Adam and longer training (15 epochs), while maintaining the lowest FLOPs and parameter count, making it computationally efficient. ResNet-34 and ResNet-50 require significantly more compute and training time, yet they do not consistently outperform ResNet-18. In fact, deeper models sometimes show reduced accuracy, likely due to optimization difficulty and overfitting on a dataset of moderate complexity. These findings suggest that for FashionMNIST, smaller architectures such as ResNet-18 provide the best balance of accuracy and efficiency, while deeper networks incur higher computational cost without proportional performance gains.