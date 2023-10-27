#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : snli_dataset.py
@author: zijun
@contact : zijun_sun@shannonai.com
@date  : 2020/11/26 14:16
@version: 1.0
@desc  : 
"""

import json
import os
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import pandas as pd

from datasets.collate_functions import collate_to_max_length


class SNLIDataset(Dataset):

    def __init__(self, directory, bert_path, max_length: int = 512):
        super().__init__()
        self.max_length = max_length
        label_map = {"SUPPORT": 0, 'REFUTED': 1, "NEI": 2}
        self.result = []
        data = pd.read_csv("/content/data.csv")
        for i in range(len(data)):
          self.result.append((data['claim'][i], data['evidence_top1'], label_map[data['verdict'][i]]))
        self.tokenizer = RobertaTokenizer.from_pretrained(bert_path)

    def __len__(self):
        return len(self.result)

    def __getitem__(self, idx):
        sentence_1, sentence_2, label = self.result[idx]
        # remove .
        if sentence_1.endswith("."):
            sentence_1 = sentence_1[:-1]
        if sentence_2.endswith("."):
            sentence_2 = sentence_2[:-1]
        sentence_1_input_ids = self.tokenizer.encode(sentence_1, add_special_tokens=False)
        sentence_2_input_ids = self.tokenizer.encode(sentence_2, add_special_tokens=False)
        input_ids = sentence_1_input_ids + [2] + sentence_2_input_ids
        if len(input_ids) > self.max_length - 2:
            input_ids = input_ids[:self.max_length - 2]
        # convert list to tensor
        length = torch.LongTensor([len(input_ids) + 2])
        input_ids = torch.LongTensor([0] + input_ids + [2])
        label = torch.LongTensor([label])
        return input_ids, label, length


def unit_test():
    root_path = "/content/snli_1.0"
    bert_path = "/content/roberta_base"
    prefix = "train"
    dataset = SNLIDataset(directory=root_path, prefix=prefix, bert_path=bert_path)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=10,
        num_workers=0,
        shuffle=False,
        collate_fn=partial(collate_to_max_length, fill_values=[1, 0, 0])
    )
    for input_ids, label, length, start_index, end_index, span_mask in dataloader:
        print(input_ids.shape)
        print(start_index.shape)
        print(end_index.shape)
        print(span_mask.shape)
        print(label.view(-1).shape)
        print()


if __name__ == '__main__':
    unit_test()
