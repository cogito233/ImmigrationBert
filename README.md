# Immigration-Bert

### A Political Speech Classification Task

The goal of this project is to finetune a transformer model that identifies political speech talks about immigration or not. 

Here we have used a standalone Distilbert base transformer model. We did not perform robust preprocessing except lightweight preprocessing. We have tried standalone Distilbert baseline with 512 maximum sequence lengths (required F1 score mention in the results section). 

#### Folder structure ####

~~~
dataloader.py       --> Used to merge the dataset based on doc id.
bert.py             --> Used to load the Distilbert model.
eval.py              --> Used to evaluate model by pandas.
main.py     
config.py          --> Used to save trainning hyperparameters.
./data             --> Contains the dataset related files.
./results           --> Contain the evaluation result based on test dataset.
~~~

#### Dataset ####

I have got two dataset dat_speeches1 contains **42540** samples and dat_speeches2 contains **34354** samples. From those data total label data distribution given below.

| type            | # of examples |
| --------------- | ------------- |
| immigration     | 1126          |
| not immigration | 874           |
| Total           | 2000          |

Then, we split the samples into train (1600), test (400) sets.

Total unseen samples without label is **74894**.

### Code usage instructions ###

First clone this repo and move to the directory. Then, install necessary libraries. Also, following commands can be used: 

~~~
$ git clone https://github.com/cogito233/ImmigrationBert.git
$ cd ImmigrationBert
$ pip install -r requirements.txt
$ huggingface-cli login
$ apt install git-lfs
$ git config --global user.email "you@example.com"
$ git config --global user.name "Your Name"
$ python main.py
~~~

### Parameters choosing ####
```
Total F1 with different epoch:
[0.8413492  0.8449074  0.83221459 0.84260191 0.84249589 0.83636347 0.8466556  0.84623581 0.84578887 0.84556199]
```

```
Learning-rate = [5e-5]
Epochs = [3]
Max seq length = [512]
Dropout = [0]
weight decay = 0.01
Batch size = [16]
```

### Results ###

Standalone Distilbert-base result base on test data.

```
               precision    recall  f1-score   support

unimmigration       0.83      0.77      0.80        91
  immigration       0.82      0.87      0.84       109

     accuracy                           0.82       200
    macro avg       0.83      0.82      0.82       200
 weighted avg       0.83      0.82      0.82       200
```

# <span style="color: red"> THE WORK IS IN PROGRESS </span>

