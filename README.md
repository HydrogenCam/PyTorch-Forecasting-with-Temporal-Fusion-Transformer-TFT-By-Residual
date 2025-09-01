# PyTorch-Forecasting with Temporal Fusion Transformer (TFT) by Residual

This repository implements **Residual Learning with Temporal Fusion Transformer (TFT)** for short-term electricity demand forecasting.  
The approach combines operator forecasts (e.g., **Total Load Forecast, TLF**) with a TFT that models **residual errors**, keeping baseline accuracy while improving **interpretability** and **responsiveness to short-term shocks**.

---

## ✨ Features
- **Residual framework**: predict residuals (`error = target − operator_forecast`) and reconstruct target (`target = operator_forecast + error_hat`).
- **Four experiment regimes**: baseline, without operator input, seasonal substitution, and residual learning.
- **Interpretability tools**: Integrated Gradients (IG), Top-k ranking, permutation and zero ablation.
- **Evaluation**: Quantile Loss during training and testing, plus visualization scripts.

---

## 📂 Repository Structure
