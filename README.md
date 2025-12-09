# Vincolo-Gradient-Stability
*A Gradient Stability Controller for Neural Network Training*

![Status](https://img.shields.io/badge/status-experimental-orange)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Stability](https://img.shields.io/badge/gradient-stability-green)
![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey)

**Vincolo** is a lightweight gradient-stability controller designed to keep neural network training stable under extreme conditions.

Unlike traditional schedulers or regularizers, Vincolo focuses on *resilience* rather than accuracy:  
it suppresses divergence at very high learning rates, mitigates shock events, and enables fast recovery from catastrophic or noisy batches.

Compatible with **LoRA**, **full fine-tuning**, and **adapter-based methods**.

