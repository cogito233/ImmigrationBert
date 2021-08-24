cfg = dict(
    MODEL_NAME = 'distilbert-base-cased',
    BATCH_SIZE = 16,
    TASK = 'immiggration',
    CHECKPOINT = '',
    LR = 5e-5,
    LABEL_LIST = ['unimmigration', 'immigration'],
)
