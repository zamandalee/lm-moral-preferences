import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
)
from preprocess import get_data_for_t5, tokenizer


class Model():
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

    def finetune_single(self, inputs):
      self.train()

      small_train_data = inputs[0:self.num_total_examples]

      for i in range(self.epochs):
          for ex in small_train_data:
              # Forward function automatically creates decoder_input_ids
              output = self.t5_model(input_ids=ex['input_ids'], lm_labels=ex['lm_labels'],
                                     attention_mask=ex['attention_mask'],
                                     decoder_attention_mask=ex['decoder_attention_mask'])
              loss = output[0]
              loss.backward()
              self.optimizer.step()
              self.optimizer.zero_grad()

          print("Epoch ", i, " ✅")

      return

    # ———————————————–  TESTING  ———————————————–

    def test_single_100(self, inputs):
        results = []

        self.eval()
        for test_ex in inputs[0:100]:
            beam_outputs = self.generate(
                input_ids=test_ex['input_ids'],
                attention_mask=test_ex['attention_mask'],
                max_length=64,
                early_stopping=True,
                num_beams=10,
                num_return_sequences=1,
                no_repeat_ngram_size=2
            )

        for beam_output in beam_outputs:
            sent = tokenizer.decode(beam_output, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True)
            # print(sent)
            results.append(sent)

        return results


def main():
    # Pre-process the data
    path = "./moral_stories_datasets/generation/action\|context/norm_distance/"
    train_data, test_data = get_data_for_t5(
        path + "train.jsonl",
        path + "test.jsonl"
    )

    # Finetune
    t5_model = Model()
    t5_model.finetune_single(train_data)

    # Test
    results = t5_model.test_single_100(test_data)

    # Save to csv
    # dict = {'base': results}
    # df = pd.DataFrame(dict)
    # df.to_csv('t5_base.csv')
