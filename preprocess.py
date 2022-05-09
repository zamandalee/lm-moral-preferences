import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow_datasets as tfds
import transformers
# import datasets
# from transformers import AutoTokenizer
import datetime
import os

# with open('moral_stories_datasets/generation/action\|context/norm_distance/train.jsonl', 'r') as handle:
#   parsed = json.load(handle)
# print(json.dumps(parsed, indent=4, sort_keys=True))

ACTION = 'action'
CONSEQUENCE = 'consequence'
NORM = 'norm'

tokenizer = transformers.AutoTokenizer.from_pretrained("t5-base")

def encode_example(example, dataset_type=ACTION, encoder_max_len=250, decoder_max_len=54):
  # For dataset_type == ACTION (action|context data)
  question = example['intention']
  norm = example['norm']
  context = example['situation']
  answer = example['moral_action']

  # if dataset_type == CONSEQUENCE:
  #   context = example['context']
  # elif dataset_type ==  NORM:
  #     context = example['context']

  question_plus = f"answer_me: {str(question)}"
  question_plus += f" norm: {str(norm)} </s>"
  question_plus += f" context: {str(context)} </s>"

  answer_plus = ', '.join([i for i in list(answer)])
  answer_plus = f"{answer_plus} </s>"

  encoder_inputs = tokenizer(question_plus, truncation=True,
                             return_tensors='tf', max_length=encoder_max_len,
                             pad_to_max_length=True)

  decoder_inputs = tokenizer(answer_plus, truncation=True,
                             return_tensors='tf', max_length=decoder_max_len,
                             pad_to_max_length=True)

  input_ids = encoder_inputs['input_ids'][0]
  input_attention = encoder_inputs['attention_mask'][0]
  target_ids = decoder_inputs['input_ids'][0]
  target_attention = decoder_inputs['attention_mask'][0]

  outputs = {'input_ids': input_ids, 'labels': target_ids,
             'attention_mask': input_attention, 'decoder_attention_mask': target_attention}
  return outputs


def generate_tf_dataset(data):
  columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']
  data.set_format(type='tensorflow', columns=columns)

  return_types = {'input_ids': tf.int32, 'attention_mask': tf.int32,
                  'labels': tf.int32, 'decoder_attention_mask': tf.int32, }
  return_shapes = {'input_ids': tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None]),
                   'labels': tf.TensorShape([None]), 'decoder_attention_mask': tf.TensorShape([None])}

  ds = tf.data.Dataset.from_generator(
      lambda: data, return_types, return_shapes)
  return ds


def get_data_for_t5(train_data_dir, test_data_dir, dataset_type=ACTION):
  train_data, test_data = [], []

  for obj in open(train_data_dir, 'r'):
    train_data.append(json.loads(obj))

  for obj in open(test_data_dir, 'r'):
    test_data.append(json.loads(obj))

  formatted_train = train_data.map(encode_example)
  formatted_test = test_data.map(encode_example)
  ex = next(iter(formatted_train))
  print("Example data from the mapped data: \n", ex)

  tf_train = generate_tf_dataset(formatted_train)
  tf_test = generate_tf_dataset(formatted_test)

  return tf_train, tf_test


def main():
    # Pre-process and tokenize the data
    path = "./moral_stories_datasets/generation/action\|context/norm_distance/"
    train_data, test_data = get_data_for_t5(
      path + "train.jsonl",
      path + "test.jsonl"
    )
