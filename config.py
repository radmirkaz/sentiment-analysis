class CFG:
    sentiment_dict = {'neutral': 1, 'neautral': 1, 'negative': 0, 'positive': 2}
    reversed_sentiment_dict = {index: gen for gen, index in sentiment_dict.items()}
    wocab_size = 219579
    MAX_SEQ_LEN = 512
    EMB_SIZE = 32
    ft_model_path = 'ft_model.model'
    saved_model_path_to_dir = 'saved_model/1'
    EPOCHS = 10
    BATCH_SIZE = 64
