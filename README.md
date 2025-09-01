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
```text
.
├── configs/                # Experiment configs (E0–E3, paths, hparams)
├── data/                   # Dataset folder (see below)
├── notebooks/              # Optional exploration notebooks
├── src/                    # Core code: datasets, models, training, evaluation
│   ├── datasets/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── scripts/                # CLI entry points (train/test/eval/plot)
├── lightning_logs/         # Saved runs (tft/version_E0, version_E1, …)
├── requirements.txt
└── README.md

## 📊 Data

The project assumes an hourly electricity dataset with **consumption**, **generation**, **price**, and **weather** features, plus an **operator forecast** column for residual learning.

**Example dataset used:**  
[📦 Kaggle – Energy Consumption, Generation, Prices and Weather (Spain, 2015–2018)](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather)

Place the CSV under `data/` or point to it with `--data_path`.

---

## 🔬 Experiments (E0–E3)

- **E0 – Baseline**  
  All variables included, TFT predicts target directly.

- **E1 – Without Operator**  
  Operator forecast removed.

- **E2 – Seasonal Substitution**  
  Operator forecast replaced by seasonal baseline.

- **E3 – Residual Learning**  
  TFT predicts residuals and reconstructs the final target.
## 🙌 Acknowledgements

- [Temporal Fusion Transformer (Lim et al., 2020)](https://arxiv.org/abs/1912.09363)  
- [PyTorch Lightning](https://www.pytorchlightning.ai/)  
- Kaggle dataset contributors



git clone https://github.com/HydrogenCam/PyTorch-Forecasting-with-Temporal-Fusion-Transformer-TFT-By-Residual
cd PyTorch-Forecasting-with-Temporal-Fusion-Transformer-TFT-By-Residual

