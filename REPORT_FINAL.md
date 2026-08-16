# REPORT - Module 3 - Assignment 3 - Deep Learning Foundations

**Name:** Netanel Farhi  
**ID:** 318590890  
**Date:** 16 August 2026  
**Chosen option:** B - Fashion-MNIST CNN

> A neural network that loses to a simpler model is a finding, not a failure. This report uses the results produced by a complete execution of the accompanying notebook.

---

## 1. Framing

The task is to classify each `28 x 28` grayscale Fashion-MNIST image into one of ten clothing classes. The primary metric is test accuracy because the classes are balanced and each error has the same assumed cost; macro F1 is included as a secondary class-balanced check.

The baseline is multinomial Logistic Regression on 784 flattened, normalized pixel features. Each pixel's absolute position is retained, but the model does not explicitly represent local neighborhoods, spatial patterns, or weight sharing. I defined DL as worth the additional cost if the CNN improved test accuracy by at least two percentage points without an unreasonable training burden; the official test set was not used for model-development decisions.

---

## 2. Results

| Model | Test metric | Parameters | Measured CPU train time | Notes |
|---|---:|---:|---:|---|
| Logistic Regression | Accuracy 0.8437; Macro F1 0.8430 | 7,850 | 353.37 s | L2 regularization; SAGA converged after 141 iterations |
| Small CNN | Accuracy 0.9145; Macro F1 0.9140 | 105,866 | 429.75 s | Two convolution blocks; dropout 0.25; validation-loss checkpointing |

The CNN improved accuracy by **7.08 percentage points**, reducing the error rate by approximately **45.3%** relative to Logistic Regression. Training times are hardware-specific; parameter counts and predictive metrics are directly comparable within this run.

### Main CNN learning-curve values

| Epoch | Train loss | Validation loss | Train accuracy | Validation accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.6134 | 0.3918 | 0.7818 | 0.8597 |
| 2 | 0.3861 | 0.3270 | 0.8632 | 0.8827 |
| 3 | 0.3358 | 0.2953 | 0.8791 | 0.8857 |
| 4 | 0.3050 | 0.2999 | 0.8909 | 0.8898 |
| 5 | 0.2865 | 0.2723 | 0.8963 | 0.8988 |
| 6 | 0.2683 | 0.2902 | 0.9032 | 0.8915 |
| 7 | 0.2522 | 0.2508 | 0.9079 | 0.9078 |
| 8 | 0.2402 | 0.2589 | 0.9119 | 0.9032 |
| 9 | 0.2284 | 0.2423 | 0.9161 | 0.9082 |
| 10 | 0.2205 | 0.2396 | 0.9184 | 0.9127 |
| 11 | 0.2112 | 0.2337 | 0.9223 | 0.9138 |
| 12 | 0.2021 | 0.2425 | 0.9255 | 0.9143 |
| 13 | **0.1938** | **0.2306** | **0.9276** | **0.9170** |
| 14 | 0.1872 | 0.2416 | 0.9301 | 0.9147 |
| 15 | 0.1793 | 0.2393 | 0.9336 | 0.9137 |

The lowest validation loss occurred at epoch 13. Training reached the 15-epoch limit before the early-stopping patience was exhausted, after which the epoch-13 checkpoint was restored for final evaluation.

---

## 3. Guiding questions

### 1. Did DL win?

Yes. The CNN reached 0.9145 test accuracy compared with 0.8437 for Logistic Regression, a gain of 7.08 percentage points, while macro F1 improved from 0.8430 to 0.9140. This exceeded the predefined two-point criterion and reduced the relative error rate by approximately 45.3%.

### 2. Logits / loss

I used `CrossEntropyLoss` because this is a mutually exclusive ten-class classification problem with integer labels. The network returns ten raw logits, and the loss applies log-softmax internally in a numerically stable way. Softmax is used only after training when probabilities are needed for interpretation.

### 3. Overfitting

Training and validation generally improved through epoch 13, where validation loss reached its minimum of 0.2306. During epochs 14 and 15, training loss continued to fall while validation loss remained above its minimum, indicating mild divergence rather than severe overfitting. The 15-epoch limit was reached before early stopping triggered, and the epoch-13 validation-loss checkpoint was restored.

### 4. Learning rate

On the controlled diagnostic subset, `lr=0.1` was too large: validation loss stayed near 2.31 and best validation accuracy was 0.103, approximately chance level. With `lr=1e-5`, validation accuracy reached only 0.6230 after four epochs, while `lr=1e-3` reached 0.8535 under the same budget. The experiment shows that an excessive learning rate can prevent learning, while a very small value makes progress unnecessarily slow.

### 5. Regularization

I used dropout at 0.25 together with validation-loss checkpointing and an early-stopping criterion. Dropout improved peak diagnostic validation accuracy by only 0.20 percentage points, but its restored checkpoint had lower validation accuracy (0.880 versus 0.884) and a slightly wider generalization gap (0.0163 versus 0.0151). Its benefit was therefore inconclusive rather than clearly positive.

### 6. Cost / benefit

Logistic Regression used 7,850 parameters and trained in 353.37 seconds, whereas the CNN used 105,866 parameters and trained in 429.75 seconds on the same CPU environment. The CNN was approximately 13.5 times larger but took only about 1.22 times as long to train in this run. In return, it gained 7.08 accuracy points, which is sufficient to justify the additional offline training cost for this task.

### 7. When DL

I would select the CNN because its accuracy gain substantially exceeded the predefined threshold. Convolution provides an appropriate spatial inductive bias: nearby pixels form reusable local patterns, whereas Logistic Regression retains absolute pixel positions but does not explicitly model locality or share features across image locations. I would retain the linear baseline when transparency, memory, or constrained edge deployment is more important than the measured gain.

### 8. Production mindset / Monday morning

I would monitor overall and per-class accuracy and F1 once labels arrive, especially the `Shirt` class, whose test F1 of 0.7361 was the weakest. I would also track input validity, image dimensions, brightness and pixel-distribution drift, predicted-class proportions, confidence, and latency. Persistent degradation, material drift, a new image source, or a changed label definition would trigger investigation and possible retraining once representative labels are available.

---

## 4. DL Model Card

### 4.1 Overview

- **Option / task / data:** Option B, ten-class Fashion-MNIST image classification with 54,000 training, 6,000 validation, and 10,000 untouched test images.
- **Architecture:** Two convolution-ReLU-max-pooling blocks, followed by a 64-unit dense layer, dropout 0.25, and ten output logits.
- **Model size:** 105,866 trainable parameters.
- **Intended use:** Educational comparison of a small CNN with a cheaper linear image baseline; not a production-ready clothing-recognition system.

### 4.2 Data and evaluation

- Images were converted to tensors and normalized with mean 0.2860 and standard deviation 0.3530.
- The original training set was split reproducibly with seed 42; the official test set remained untouched until final evaluation.
- Primary metric: accuracy. Secondary checks: macro F1, per-class precision/recall/F1, confusion matrix, and the first 12 misclassified test images.
- The strongest class by F1 was `Bag` at 0.9859; the weakest was `Shirt` at 0.7361.

### 4.3 Training setup

- **Loss:** `CrossEntropyLoss` on raw logits.
- **Optimizer:** Adam with learning rate `1e-3`.
- **Regularization:** Dropout 0.25, validation-loss checkpointing, and an early-stopping patience of 3.
- **Checkpoint:** Training reached the 15-epoch limit before early stopping triggered; the epoch-13 checkpoint was restored.
- **Reproducibility:** Python, NumPy, PyTorch, split, and DataLoader seeds were fixed at 42; CUDA deterministic settings are enabled when CUDA is available.

### 4.4 Performance and cost / benefit

| Dimension | Logistic Regression | Small CNN |
|---|---:|---:|
| Test accuracy | 0.8437 | **0.9145** |
| Test macro F1 | 0.8430 | **0.9140** |
| Parameters | **7,850** | 105,866 |
| Measured CPU train time | **353.37 s** | 429.75 s |
| Image structure | Absolute positions retained; locality not explicitly modeled | Explicitly exploited |
| Interpretability | Direct pixel coefficients | Distributed learned features |

The CNN was approximately 13.5 times larger and took about 1.22 times longer to train in this run, but improved accuracy by 7.08 points. The linear baseline remains attractive when simplicity, transparency, memory, or deployment constraints dominate.

### 4.5 Diagnostics and limitations

- The training curves show mild divergence after the epoch-13 minimum validation loss, and the epoch-13 checkpoint is restored.
- `lr=0.1` failed to learn, `1e-5` learned too slowly, and `1e-3` gave the strongest progress under the equal four-epoch diagnostic budget.
- Dropout's measured benefit was inconclusive: peak validation accuracy improved marginally, but the restored checkpoint did not generalize better.
- Fashion-MNIST is clean, centered, low-resolution, and balanced. Results should not be assumed to transfer to photographs, rotated or occluded products, imbalanced categories, or classes outside the ten-label taxonomy.
- Training time varies by hardware, software version, and device, so it should not be treated as a universal benchmark.

### 4.6 Deployment decision and monitoring

I would select the CNN because its 7.08-point gain is substantial and exceeds the predefined decision threshold. Production monitoring would cover delayed-label performance by class, image and prediction drift, confidence, input validity, latency, and class prevalence. Persistent degradation, a new data source, or a changed taxonomy would trigger review and comparison against the current champion after representative labels become available.

---

## 5. Reflection

The largest surprise was not that the CNN won, but how completely the learning rate of 0.1 prevented learning: a plausible-looking optimization choice left the network at chance performance. The class-level analysis was also important because 91.45% overall accuracy hides the much weaker `Shirt` F1 of 0.7361.

For my pairs-trading mid-term project, I would not move to DL only because it is available. The engineered tabular event features and limited number of independent market events favor simpler models; DL would become reasonable only with a much larger sequential dataset, raw price and volume histories, and a time-aware architecture that produces a verified out-of-sample improvement after transaction costs.
