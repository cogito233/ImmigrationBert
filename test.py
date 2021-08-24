from dataloader import DataLoader, DataFilter
from eval import EvalClass
from bert import load_final_model
from icecream import ic
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from datasets import load_dataset
from utils import set_seed
from config import cfg

import numpy as np
import pandas as pd

label_list = cfg['LABEL_LIST']

def test():
    model = load_final_model()
    dataset = DataLoader(cfg, mode = '5-cross-inference')
    args =  TrainingArguments(
            f"model-5-cross-{cfg['TASK']}-final",
            evaluation_strategy = "epoch",
            per_device_train_batch_size=cfg['BATCH_SIZE'],
            per_device_eval_batch_size=cfg['BATCH_SIZE'],
            num_train_epochs=3,
            learning_rate = cfg['LR'],
            weight_decay=0.01,
        )
    evalClass = EvalClass() 
    trainer = Trainer(model, args, train_dataset = dataset.train, eval_dataset = dataset.train,compute_metrics=evalClass.compute_metrics)
    print(trainer.evaluate())
# Test the output model

if __name__=='__main__':
    test()