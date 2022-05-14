# Eliciting and Shaping Moral Preferences of Language Models

This is the CS2952N final project by Mason Zhang, William Yang, and Amanda Lee.

*Project Outline*:
- `preprocess.py`: tokenization of dataset (note: segregation of dataset into the action|context+norm, etc structres is not included here)
- `t5_model.py`: fine-tuning of the T5 model without batching. We used this for early testing purposes, then reverted to it when we encountered challenges regarding GCP and large-scale training
- `t5_model_batched.py`: fine-tuning, with batching

*To Run*:
Note: best to run on Google Colab file in the code base.
1. Install `requirements.txt`
2. Uncomment the csv download lines in `model.py`'s `main()`
3. Run `model.py`

*Acknowledgements*:
We used the tutorial on fine-tuning T5 using Tensorflow to reinforce our understanding the model, and get a basic idea of training/testing practices.