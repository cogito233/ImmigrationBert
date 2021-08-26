from transformers import AutoModelForSequenceClassification
import torch
import logging

def build_model(label_list, cfg):
    model = AutoModelForSequenceClassification.from_pretrained(cfg['MODEL_NAME'], num_labels=len(label_list))
    return model

def save_model(model, idx):
    PATH = f"./model/model_params_{idx}.pth"
    torch.save(model.state_dict(), PATH)

def  load_model(model, idx):
    PATH = f"./model/model_params_{idx}.pth"
    model.load_state_dict(torch.load(PATH))

def save_final_model(model):
    PATH = f"./model/final_model.pkl"
    torch.save(model, PATH)

def load_final_model():
    PATH = f"./model/final_model.pkl"
    return torch.load(PATH)

def save_with_huggingface_format(model, cfg):
    logging.info(f"Saved model to {cfg['SAVE_DIR']}")
    model.save_pretrained(cfg['SAVE_DIR'])

def load_with_huggingface_format(cfg):
    logging.info("Restoring model from {}".format(cfg['SAVE_DIR']))
    model = AutoModelForSequenceClassification.from_pretrained(cfg['SAVE_DIR'])
    model.to(cfg['DEVICE'])
    return model