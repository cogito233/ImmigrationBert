import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
from config import cfg

label_list = cfg['LABEL_LIST']
# The reason why set the class is to record the history of the metrics
class EvalClass(object):
    def __init__(self):
        self.acc = []
        self.f1 = []

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=-1).tolist()
        print(classification_report( [label_list[x] for x in predictions], [label_list[x] for x in labels], labels=label_list))
        self.acc.append(accuracy_score(predictions, labels))
        self.f1.append( f1_score(predictions, labels))
        results = dict(
            confusion_matrix = confusion_matrix(predictions, labels).tolist(), 
            accuracy_score = accuracy_score(predictions, labels), 
            f1_score = f1_score(predictions, labels), 
            precision_score = precision_score(predictions, labels), 
            recall_score = recall_score(predictions, labels), 
        )
        return results

def test():
    x=EvalClass()
    x.compute_metrics(([[0,1]]*200,[0,1]*100))
    print("###################################################")
    print(x.acc)

if __name__=='__main__':
    test()