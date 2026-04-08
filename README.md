# Sticker Personalization Pipeline

## Dataset
Download the following datasets:

- https://huggingface.co/datasets/metchee/sticker-queries/
- https://huggingface.co/datasets/metchee/u-sticker

Use these two datasets to construct the training data:

- Align sticker IDs between both datasets  
- Combine user interaction history with sticker query annotations  
- Build training samples in the form of:
  `(user_history, query, sticker)`
---

## LoRA Training

For LoRA training, please refer to:  
https://github.com/cloneofsimo/lora

---

## Pipeline

Run the following scripts in order:

```bash
python3 lora_prep.py
python3 style_prediction.py
python3 query_adaptation.py
```
`lora_prep.py`: prepares and formats the dataset for training
`style_prediction.py`: extracts user style representations
`query_adaptation.py`: refines queries into model-friendly prompts

```
python3 train.py
```
