# 📄 Document Image Understanding & Analysis

> Fine-tuning transformer models (BERT, RoBERTa, LayoutLM, GPT-2) for document understanding and token classification on structured document datasets.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-orange?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🎯 Objective

This project explores **Document Image Understanding** — the task of classifying tokens in scanned documents into semantic categories:

| Label | Description |
|---|---|
| `Answer` | Answer fields in forms |
| `Question` | Question fields in forms |
| `Header` | Document headers |
| `Other` | Other content |
| `PAD` | Padding tokens |

The goal: benchmark multiple transformer architectures on this structured document classification task.

---

## 🛠️ Models & Stack

| Component | Technology |
|---|---|
| Base Models | BERT, RoBERTa, LayoutLM, GPT-2 |
| Framework | HuggingFace Transformers |
| Deep Learning | PyTorch |
| Environment | Google Colab |

---

## 🗂️ Project Structure

```
Document-Image-Understanding-and-Analysis/
│
├── 📓 LayoutLM.ipynb         ← LayoutLM fine-tuning (layout-aware)
├── 🐍 Bert.py                ← BERT experiments
├── 🐍 Roberta.py             ← RoBERTa experiments
├── 🐍 Layout-LM.py           ← LayoutLM script
├── 📄 GPT-2                  ← GPT-2 experiment
└── 📖 README.md
```

---

## 📊 Experimental Results

### BERT — Token Classification

> 7 experiments varying epochs and learning rate

| Experiment | Epochs | LR | Accuracy | Best F1 (Answer) |
|---|---|---|---|---|
| Exp 1 | — | random init | 0.4106 | 0.5110 |
| Exp 2 | 3 | 3e-5 | 0.4189 | 0.5302 |
| Exp 3 | 5 | 3e-5 | 0.4369 | 0.5219 |
| Exp 4 | 5 | 2e-5 | 0.4042 | 0.5041 |
| Exp 5 | 5 | 2e-5 | 0.4042 | 0.5041 |
| Exp 6 | 5 | 2e-5 | 0.4042 | 0.5041 |
| Exp 7 | 5 | 2e-5 | 0.4186 | 0.5122 |

**Key insight:** BERT struggles with `Header` classification (F1 ≈ 0) across all experiments — suggesting the model lacks layout awareness to distinguish headers from body text.

---

### RoBERTa — Token Classification

> 4 experiments varying epochs (3 → 20)

| Experiment | Epochs | Best Accuracy | Best F1 |
|---|---|---|---|
| Exp 1 | 3 | 0.7894 | 0.2020 |
| Exp 2 | 5 | 0.7993 | 0.2348 |
| Exp 3 | 7 | 0.7970 | 0.2272 |
| Exp 3b | 10 | 0.7997 | 0.2365 |
| Exp 4 | 20 | 0.7894 | 0.3136 |

**Key insight:** RoBERTa achieves higher accuracy than BERT but F1 remains low — the model converges quickly and then overfits. More epochs don't help after epoch 5.

---

## 💡 Key Takeaways

- **BERT** with random parameters already learns `Answer` tokens reasonably well (F1 ~0.51) but completely fails on `Header` — a structural label that requires layout context
- **RoBERTa** shows better raw accuracy but similar F1 ceiling — text-only models have inherent limits on document understanding tasks
- **LayoutLM** (layout-aware) is the natural next step — it incorporates bounding box coordinates alongside text, making it purpose-built for this task
- Optimal learning rate appears to be around **2e-5 to 3e-5** across both architectures

---

## 🔬 Context

This project was conducted as part of my graduate coursework in NLP and representation learning — benchmarking transformer architectures before the emergence of layout-aware models as the standard for document AI.

It connects directly to my later work on:
- 🏥 [RAG Chatbot](https://github.com/samibahig/RAG_Chatbot) — grounding LLM responses in documents
- 🩺 [Protocol Imaging Classification](https://github.com/samibahig/Prediction-Image-Protocole-) — applied document understanding in healthcare

---

## 👤 Author

**Sami Bahig** — Data Scientist & AI Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/samibahig)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/samibahig)

---

*MIT License · Sami Bahig · 2023*
