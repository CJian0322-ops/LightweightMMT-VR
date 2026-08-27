# Lightweight MMT for 360° VR Head Prediction

PyTorch implementation of the Lightweight Multi-Modal Transformer from our MSys2026 paper.

Predicts 1.0s head trajectory from 2.0s past rotation + 15-D visual saliency features. Around0.85M params, ~8.6ms inference on Snapdragon XR2.

## What's in this repo
- `inference.py` — model + prediction function
- `model_weights.pt` — pretrained checkpoint
- `example.py` — minimal demo with random data## What's NOT here
Training code and the original VR datasets aren't included — too if you are interested in training, drop me an email (Jian0322@163.com).

## Run it
