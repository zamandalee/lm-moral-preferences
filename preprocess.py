# COLAB LINK: https://colab.research.google.com/drive/1MIiMknu7tqp55l9V5RHRPcBHboJXtMbc?authuser=1#scrollTo=pMRXAmMyJUC-

import json
# import tensorflow as tf
import transformers
from transformers import AutoTokenizer

# Target (model output) options
ACTION = 'action'
CONSEQUENCE = 'consequence'
NORM = 'norm'


# ———————————————–  PREPROCESSING  ———————————————–

tokenizer = transformers.AutoTokenizer.from_pretrained("t5-base")

def encode_example(example, encoder_max_len=250, decoder_max_len=54):
    # For dataset_type == ACTION (action|context data)
    # Format intention, norm, situation, and action
    question = example['intention']
    norm = example['norm']
    context = example['situation']
    answer = example['moral_action']

    # into question and answer
    question_plus = f"{str(norm)}"
    question_plus += f" {str(context)} </s>"
    q = f"{str(question)}"
    answer_plus = f"{answer} </s>"

    # Tokenize
    encoder_inputs = self.tokenizer.encode_plus(
        question_plus, max_length=encoder_max_len,
        pad_to_max_length=True, return_tensors="pt"
    )
    dtokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    decoder_inputs = dtokenizer.encode(
        q,
        question_plus,
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
    )

    # Return necessary args for t5 model fine-tuning
    input_ids = encoder_inputs["input_ids"]
    input_attention = encoder_inputs["attention_mask"]
    target_ids = decoder_inputs["input_ids"]
    target_attention = decoder_inputs["attention_mask"]

    outputs = {'question_plus': question_plus, 'answer_plus': answer_plus,
              'input_ids': input_ids, 'lm_labels': target_ids,
              'attention_mask': input_attention,
              'decoder_attention_mask': target_attention}
    return outputs


# Legacy from when we were using TF:
# def generate_tf_dataset(data):
#   columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']
#   data.set_format(type='tensorflow', columns=columns)

#   return_types = {'input_ids': tf.int32, 'attention_mask': tf.int32,
#                   'labels': tf.int32, 'decoder_attention_mask': tf.int32, }
#   return_shapes = {'input_ids': tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None]),
#                    'labels': tf.TensorShape([None]), 'decoder_attention_mask': tf.TensorShape([None])}

#   ds = tf.data.Dataset.from_generator(
#       lambda: data, return_types, return_shapes)
#   return ds


def get_data_for_t5(train_data_dir, test_data_dir, dataset_type=ACTION):
  # Load in the dataset
  data_dir = ""
  train_data_dir = data_dir + "train.jsonl"
  test_data_dir = data_dir + "test.jsonl"

  og_train_data, og_test_data = [], []
  for obj in open(train_data_dir, 'r'):
      og_train_data.append(json.loads(obj))
  for obj in open(test_data_dir, 'r'):
      og_test_data.append(json.loads(obj))

  og_train_data = list(filter(lambda x: x['label'] == '1', og_train_data))
  og_test_data = list(filter(lambda x: x['label'] == '1', og_test_data))

  print("Train, test data lengths: ", len(og_train_data), len(og_test_data))
  print("Example original data: \n", og_train_data[0])

  train_data = list(map(encode_example, og_train_data))
  test_data = list(map(encode_example, og_test_data))

  return train_data, test_data

