from transformers import AutoModelForSequenceClassification

def build_model(label_list, cfg):
    model = AutoModelForSequenceClassification.from_pretrained(cfg['MODEL_NAME'], num_labels=len(label_list))
    return model