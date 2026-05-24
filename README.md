# Automatic Medical Report Generation from Chest X-Ray Images

> Deep Learning · Rice University COMP576 · Aug – Dec 2023

Automated radiology report generation using a multi-modal neural network that combines chest X-ray image features with clinical language models. The system accepts raw X-ray images and produces descriptive diagnostic reports — reducing radiologist workload and improving consistency.

## Highlights
- **30% accuracy improvement** over baseline encoder-decoder models
- Novel **co-attention mechanism** that jointly attends to visual regions and text tokens
- Integrates **CheXNet** (pre-trained on CheXpert), **GloVe** embeddings, and **BioBERT** for clinical NLP
- Led a **team of 5** researchers end-to-end: data preprocessing → training → evaluation

## Architecture
```
Chest X-Ray Image
       │
   CheXNet (DenseNet-121)        ← Pre-trained on CheXpert
       │
  Visual Feature Vectors
       │
  Co-Attention Module  ←──────── GloVe / BioBERT Text Embeddings
       │
  Encoder–Decoder (LSTM)
       │
  Generated Report Text
```

## Results
| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|
| Baseline Encoder-Decoder | 0.31 | 0.20 | 0.14 | 0.10 |
| **Ours (Co-Attention)** | **0.40** | **0.26** | **0.18** | **0.13** |

## Setup

### Prerequisites
```bash
pip install torch torchvision transformers nltk Pillow pandas numpy
```
1. Download [CheXpert Images](https://academictorrents.com/details/5a3a439df24931f410fac269b87b050203d9467d) & [Reports](https://academictorrents.com/details/66450ba52ba3f83fbf82ef9c91f2bde0e845aba9)
2. Download CheXNet pre-trained [weights](https://drive.google.com/file/d/19BllaOvs2x5PLV_vlWMy4i8LapLb2j6b/view)
3. Install Jupyter Lab: `pip install jupyterlab`

### Running the Code
```bash
# Step 1 — Preprocess dataset
jupyter nbconvert --to notebook --execute "Preprocessing Data.ipynb"

# Step 2 — Extract image features via CheXNet
jupyter nbconvert --to notebook --execute "ChexNet_Enc_Dec.ipynb"

# Step 3 — Train the model (choose one)
jupyter nbconvert --to notebook --execute "Encoder_Decoder.ipynb"
# or with co-attention:
jupyter nbconvert --to notebook --execute "Attention_Model.ipynb"
```

## Research Poster
![Poster](./Poster.png)

## Tech Stack
`Python` `PyTorch` `CheXNet` `BioBERT` `GloVe` `NLTK` `Flask` `Jupyter`
