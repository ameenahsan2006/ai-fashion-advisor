# 👗 AI Personal Fashion Stylist Studio (v5.0 Editorial Luxury Edition)

[![Google Colab](https://img.shields.io/badge/Google%20Colab-Runnable-orange?logo=googlecolab)](https://colab.research.google.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![timm EfficientNet](https://img.shields.io/badge/Backbone-EfficientNet_B3_timm-blue?logo=pytorch)](https://github.com/huggingface/pytorch-image-models)
[![UI](https://img.shields.io/badge/UI-Editorial_Luxury_Atelier-purple?logo=gradio)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Overview
The **AI Personal Fashion Stylist Studio v5.0** is an enterprise-grade intelligent fashion consultation system built with an **Editorial Luxury Fashion-Brand Aesthetic**. Given an image of any garment or footwear (Tops, Pants, Footwear, or Full Outfits), it predicts the category with high precision using an **EfficientNet-B3 (`timm`)** backbone and generates a **18-section personal styling consultation dossier** with bi-directional matching logic, color psychology, and visual feature explainability.

---

## 🌟 Editorial Luxury Frontend Features (v5.0)

1. **Dark Gradient Hero Overlay**: Full-bleed dark slate backdrop (`#0b0f19` to `#1e293b`) with radial lighting highlights.
2. **Editorial Display Typography**: Uses Google Font **Cormorant Garamond** for magazine-style headers, featuring an **italic warm amber accent word** (*Tailored*) and **all-caps warm rose eyebrow labels**.
3. **Pre-Interaction Idle State**: Friendly empty state card ("*Your Styling Dossier Awaits*") displayed prior to image upload.
4. **Micro-Motion Entrance Animations**: Smooth `@keyframes fadeInUp` slide-in keyframe animations with staggered delays (`0.05s` to `0.4s`).
5. **Bi-Directional Consultation Cards**: Dynamically adapts advisory cards based on whether the input is Topwear, Bottomwear (Pants), Footwear, or a Full Outfit.

---

## 🚀 Quickstart Guide (Google Colab)

1. Open [`AI_Fashion_Advisor.ipynb`](./AI_Fashion_Advisor.ipynb) in **Google Colab**.
2. Set Runtime accelerator: **Runtime -> Change runtime type -> T4 GPU**.
3. Click **Runtime -> Run all** (`Ctrl + F9`).
4. `kagglehub` downloads the dataset automatically in <20 seconds.
5. The model trains in ~2.5 minutes, launching your v5.0 Editorial Luxury AI Personal Fashion Stylist Studio!
