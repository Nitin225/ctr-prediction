| Model | AUC | Log loss | ECE | Fit time (s) |
|---|---|---|---|---|
| Constant (train click rate) | 0.5000 | 0.5633 | - | - |
| Logistic Regression | 0.7809 | 0.4623 | 0.0029 | 45 |
| XGBoost (integer codes) | 0.7784 | 0.4640 | 0.0074 | 17 |
| XGBoost (+ frequency encoding) | 0.7790 | 0.4635 | 0.0071 | 28 |
| DeepFM (PyTorch) | 0.7852 | 0.4598 | 0.0135 | 84 |
