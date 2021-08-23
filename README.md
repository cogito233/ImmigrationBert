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

Then, we split the samples into train (1600), validation (200), test (200) sets.

Total unseen samples without label is **74894**.

### Code usage instructions ###

First clone this repo and move to the directory. Then, install necessary libraries. Also, following commands can be used: 

~~~
$ git clone https://github.com/cogito233/ImmigrationBert.git
$ cd ImmigrationBert
$ python main.py
~~~

### Parameters choosing ####

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
# TODO
```

# <span style="color: red"> THE WORK IS IN PROGRESS </span>

