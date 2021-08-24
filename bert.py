from transformers import AutoModelForSequenceClassification
import torch

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