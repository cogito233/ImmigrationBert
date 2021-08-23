from config import cfg
from dataloader import DataLoader, DataFilter
from eval import compute_metrics
from bert import build_model
from icecream import ic
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from datasets import load_dataset

import numpy as np
import pandas as pd

label_list = cfg['LABEL_LIST']

def test_model():
    dataset = DataLoader(cfg)
    # tokenizer = AutoTokenizer.from_pretrained(cfg['MODEL_NAME'])
    # data_collator = DataCollatorForTokenClassification(tokenizer)
    args =  TrainingArguments(
        f"test-{cfg['TASK']}",
        evaluation_strategy = "epoch",
        per_device_train_batch_size=cfg['BATCH_SIZE'],
        per_device_eval_batch_size=cfg['BATCH_SIZE'],
        num_train_epochs=3,
        weight_decay=0.01,
    )

    for i in range(5):
        model = build_model(label_list, cfg)
        trainer = Trainer(
            model,
            args,
            train_dataset = dataset.train[i],
            eval_dataset = dataset.test[i],
            compute_metrics=compute_metrics
        )
        """    data_collator = data_collator,
            tokenizer = tokenizer,
            
        )"""
        trainer.train()
        trainer.evaluate()

def train_final_model():
    model = build_model(label_list, cfg)
    dataset = DataLoader(cfg, mode = 'inference')
    # tokenizer = AutoTokenizer.from_pretrained(cfg['MODEL_NAME'])
    # data_collator = DataCollatorForTokenClassification(tokenizer)
    args =  TrainingArguments(
        f"test-{cfg['TASK']}-final",
        per_device_train_batch_size=cfg['BATCH_SIZE'],
        per_device_eval_batch_size=cfg['BATCH_SIZE'],
        num_train_epochs=3,
        weight_decay=0.01
    )
    trainer = Trainer(model, args, train_dataset = dataset.train)
    trainer.train()
    processRawData(trainer)
    # 保存预测结果

def processRawData(trainer):
    tokenizer = AutoTokenizer.from_pretrained(cfg['MODEL_NAME'])
    def filterRawData(example):
        tokenized_inputs = tokenizer(example['text'],padding="max_length" , truncation=True)
        tokenized_inputs["label"] = 0
        tokenized_inputs['doc_id'] = example['doc_id']
        tokenized_inputs['speech_id'] = example['speech_id']
        return tokenized_inputs
    
    dataset1 = load_dataset('csv', data_files='./data/dat_speeches_043114_immi_h_ascii_07212021.csv')
    dataset2 = load_dataset('csv', data_files='./data/dat_speeches_043114_immi_s_ascii_07202021.csv')
    dataset1 = dataset1.map(filterRawData, batched = False)
    dataset2 = dataset2.map(filterRawData, batched = False)
    Calc_and_Save(trainer, dataset1['train'], 'hand_coding_task_house_all_08172021_lite.csv ')
    Calc_and_Save(trainer, dataset2['train'], 'hand_coding_task_senate_all_08172021_lite.csv ')

def Calc_and_Save(trainer, dataset, outdir):
    predictions, labels, _ = trainer.predict(dataset)
    predictions = np.argmax(predictions, axis=-1).tolist()
    data = [[dataset[i]['doc_id'], dataset[i]['speech_id'], predictions[i]] for i in range(len(predictions))]
    df = pd.DataFrame(data,columns = ['doc_id','speech_id','immigration'])
    df.to_csv(outdir,sep = ',',index=False) 

if __name__=='__main__':
    train_final_model()