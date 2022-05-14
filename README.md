# Eliciting and Shaping Moral Preferences of Language Models

This is the CS2952N final project by Mason Zhang, William Yang, and Amanda Lee.

*Project Outline*:
- `preprocess.py`: tokenization of dataset (note: segregation of dataset into the action|context+norm, etc structres is not included here)
- `model.py`: fine-tuning of the T5 model without batching. We used this for early testing purposes, then reverted to it when we encountered challenges regarding GCP and large-scale training
- `model_batched.py`: fine-tuning, with batching

*To Run*:
1. Install `requirements.txt`
2. Run `model.py`