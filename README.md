# KAN PDE Thesis Code

Implementation accompanying the Master's Thesis
**“Kolmogorov–Arnold Theorem and Its Applications to Neural Networks”**,
University of Calabria.

This repository provides the implementation developed for my Master's Thesis in Mathematics.

The project implements a numerical solver for the **1D Fokker–Planck equation** using **Kolmogorov–Arnold Networks (KAN)** within a **Dual Physics-Informed Neural Network (Dual-PINN)** framework.

---

## Author

**Ugo Giorgio Samuele Campolo**
University of Calabria

---

## Description

The implementation includes:

* Kolmogorov–Arnold Network (KAN) architecture
* Dual Physics-Informed Neural Network training strategy
* Boundary–interior specialization
* Numerical solution of the 1D Fokker–Planck partial differential equation

This project accompanies the research work developed during the Master's Thesis in Mathematics.

---

## Results

Example result for the 1D Fokker–Planck equation:

![Fokker-Planck Result](results/fokker_planck_best_seed48.png)



Best model performance:

- Relative L2 error: 0.027
- L2 accuracy: 97.3%
- Total parameters: 160

---

## Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Run

Execute the main experiment:

```bash
python kan_doublepinn.py
```

---

## References

[1] Abbas, N., Colao, V., Macrì, D., Spataro, W. (2025).
*A Multi-Phase Dual-PINN Framework: Soft Boundary–Interior Specialization via Distance-Weighted Priors*.
arXiv preprint arXiv:2511.23409.

[2] LeBleu, B. (2024).
*efficient-KAN: An Efficient Implementation of Kolmogorov–Arnold Networks*.
GitHub Repository.
https://github.com/Blealtan/efficient-kan

---

## Thesis

📄 **Kolmogorov–Arnold Theorem and Its Applications to Neural Networks**

The full Master's Thesis accompanying this repository is available here:

[Download Thesis PDF](thesis/Campolo_Kolmogorov_Arnold_Theorem_Neural_Networks.pdf)

