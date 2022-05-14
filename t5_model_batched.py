import json
import numpy as np
import pandas as pd
import random
import torch
import matplotlib.pyplot as plt
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from preprocess import get_data_for_t5, tokenizer


class Batched_Model():
    def __init__(self):
        epochs = 5
        batch_size = 240
        num_total_examples = 2400

        t5_model = T5ForConditionalGeneration.from_pretrained(
            't5-large')  # Hugging Face pre-trained model

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in t5_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in t5_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=3e-4, eps=1e-8)

    # ———————————————–  FINE-TUNING  ———————————————–
    def preprocess(data, encoder_max_len=250, decoder_max_len=54):
        input_ids = []
        masks = []
        lm_labels = []
        decoder_masks = []

        question_pluses = []
        answers_pluses = []

        for i, example in enumerate(data):
            # For dataset_type == ACTION (action|context data)
            # Format intention, norm, situation, and action
            question = example['intention']
            norm = example['norm']
            context = example['situation']
            answer = example['moral_action']

            # into question and answer
            question_plus = f"answer_me: {str(question)}"
            question_plus += f" norm: {str(norm)}"
            question_plus += f" context: {str(context)} </s>"
            answer_plus = f"{answer} </s>"

            question_pluses.append(question_plus)
            answers_pluses.append(answer_plus)

            # Tokenize
            encoder_inputs = tokenizer.encode_plus(
                question_plus, max_length=encoder_max_len,
                pad_to_max_length=True, return_tensors="pt"
            )
            decoder_inputs = tokenizer.encode_plus(
                answer_plus, max_length=decoder_max_len,
                pad_to_max_length=True, return_tensors="pt"
            )

            input_ids.append(encoder_inputs["input_ids"])
            masks.append(encoder_inputs["attention_mask"])
            lm_labels.append(decoder_inputs["input_ids"])
            decoder_masks.append(decoder_inputs["attention_mask"])

        return (input_ids, masks, lm_labels, decoder_masks, question_pluses, answers_pluses)


    # ———————————————–  FINE-TUNING  ———————————————–

    def set_seed(seed):
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)

    def train(self, train_data):
        # Train the model
        self.t5_model.train()

        for i, start_idx in enumerate(range(0, self.num_total_examples, self.batch_size)):
            # Batch
            end_idx = start_idx + self.batch_size
            batch_inputs = train_data[start_idx:end_idx]

            (input_ids, masks, lm_labels, decoder_masks, _, _) = train_data

            # input_ids = list(map(x['input_ids']))
            # lm_labels = list(map(x['lm_labels'] for x in batch_inputs))
            # attention_mask = list(map(x['attention_mask'] for x in batch_inputs))
            # decoder_attention_mask = list(
            #     map(x['decoder_attention_mask'] for x in batch_inputs))

            # Forward function automatically creates decoder_input_ids
            output = self.t5_model(input_ids=input_ids, lm_labels=lm_labels,
                                   attention_mask=masks,
                                   decoder_attention_mask=decoder_masks)
            loss = output[0]
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if (i % 100 == 0):
                print("Example ", i, " ✅")


    # ———————————————–  TESTING  ———————————————–

    def test(self, test_data):
        self.t5_model.eval()

        (input_ids, masks, _, _, _, _) = test_data

        results = []
        beam_outputs = self.t5_model.generate(
            input_ids=input_ids,
            attention_mask=masks,
            max_length=64,
            early_stopping=True,
            num_beams=10,
            num_return_sequences=1,
            no_repeat_ngram_size=2
        )

        for beam_output in beam_outputs:
            sent = tokenizer.decode(beam_output, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True)
            results.append(sent)
            # print(sent)


def main():
    # Pre-process the data
    path = "./moral_stories_datasets/generation/action\|context/norm_distance/"

    og_train_data, og_test_data = [], []

    for obj in open(path + "train.jsonl", 'r'):
        og_train_data.append(json.loads(obj))
    for obj in open(path + "test.jsonl", 'r'):
        og_test_data.append(json.loads(obj))

    og_train_data = list(filter(lambda x: x['label'] == '1', og_train_data))
    og_test_data = list(filter(lambda x: x['label'] == '1', og_test_data))

    t5_model = Batched_Model()

    train_data = Batched_Model.preprocess( og_train_data)
    test_data = Batched_Model.preprocess(og_test_data)

    # Finetune
    t5_model.set_seed(42)
    t5_model.train(train_data)

    # Test
    results = t5_model.test(test_data)

    # Save to csv
    # dict = {'base': results}
    # df = pd.DataFrame(dict)
    # df.to_csv('t5_base.csv')
