import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
from config import cfg

label_list = cfg['LABEL_LIST']
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1).tolist()
    print(classification_report( [label_list[x] for x in predictions], [label_list[x] for x in labels], labels=label_list))
    results = dict(
        confusion_matrix = confusion_matrix(predictions, labels).tolist(), 
        accuracy_score = accuracy_score(predictions, labels), 
        f1_score = f1_score(predictions, labels), 
        precision_score = precision_score(predictions, labels), 
        recall_score = recall_score(predictions, labels), 
    )
    return results

def test():
    compute_metrics(([[0,1]]*200,[0,1]*100))

if __name__=='__main__':
    test()