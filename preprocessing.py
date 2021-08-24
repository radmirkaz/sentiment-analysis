from gensim.models.fasttext import FastText
from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd
import numpy as np

import nltk
import string
import tqdm
import pymorphy2
from config import CFG
import argparse
nltk.download('punkt')
nltk.download('stopwords')


def tokenize_text(raw_text: str, stop_words, morph):
    tokenized_str = nltk.word_tokenize(raw_text)
    tokens = [i.lower() for i in tokenized_str if (i not in string.punctuation+'...'+'``'+"''"+"»"+'–'+'—')]
    filtered_tokens = [i for i in tokens if ( i not in stop_words )]
    lemmatized_tokens = [morph.parse(i)[0].normal_form for i in filtered_tokens]
    return lemmatized_tokens


def text2int(data):
    """преобразования описания в последовательность чисел"""
    words_dict = {'<PAD>': 0}
    words_dict['<UNK>'] = 1

    index = 2
    for seq in tqdm.tqdm(data['tokenized']):
        for token in seq:
            if token not in words_dict:
                words_dict[token] = index
                index += 1
    print('\nVocabulary length: ', index)

    return words_dict


# последовательность индексов в строку из токенов-слов
def decode_text(text, inverse_words_dict):
    return ' '.join([inverse_words_dict.get(i, '?') for i in text])


# текст в последовательность индексов
def encode_text(text, words_dict):
    words = tokenize_text(text)
    idxs = [words_dict.get(word, words_dict['<UNK>']) for word in words]
    return idxs


def encode_to_vector(tokens, ft_model):
    """векторизация"""

    vectors = []
    for token in tokens:
        try:
            vectors.append(ft_model.wv.get_vector(token))
        except KeyError:
            # skip unknown tokens
            continue
    return vectors


def padding_sequence(list_of_vectors, MAX_SEQ_LEN, EMB_SIZE):
    """
    преобразовать список векторов в массив формы (MAX_SEQ_LEN, EMB_SIZE), добавив 0 к коротким предложениям (дополнение сообщения)
    или обрезка до MAX_SEQ_LEN для длинных предложений
    к концу, размер вывода функции равен (1, MAX_SEQ_LEN, EMB_SIZE)
    """

    sequence = np.array(list_of_vectors)
    if sequence.shape[0] < MAX_SEQ_LEN and sequence.shape[0] > 0:
        sequence = np.vstack([sequence, np.zeros(shape=(MAX_SEQ_LEN - sequence.shape[0], sequence.shape[1]))])
    elif sequence.shape[0] > MAX_SEQ_LEN:
        sequence = sequence[:MAX_SEQ_LEN]
    elif sequence.shape[0] == 0:
        sequence = np.ones((MAX_SEQ_LEN, EMB_SIZE))
    assert sequence.shape == (MAX_SEQ_LEN,
                              EMB_SIZE), f'padding_sequence output is incorrect: {sequence.shape} instead of {(MAX_SEQ_LEN, EMB_SIZE)}'

    return sequence[np.newaxis, ...].astype(np.float32)


def tensor_batch_generator(data, ft_model, MAX_SEQ_LEN, EMB_SIZE, batch_size=1024):
    """
    Генератор используется для экономии памяти во время векторизации наборов данных трейна / тест.
    data - pd.Series (или другой итеративный тип) со списками токенов, например: df ['tokenized']
    """

    length = len(data)
    for i in range(0, length, batch_size):
        X = []
        for tokens in data[i:min(i + batch_size, length)]:
            X.append(padding_sequence(encode_to_vector(tokens, ft_model), MAX_SEQ_LEN, EMB_SIZE))
        yield np.vstack(X)


def create_prepared_data(data, ft_model, MAX_SEQ_LEN, EMB_SIZE, batches_size=1014):
    """
    Создать массив векторизованных последовательностей с правым заполнением из токенизированного текста
    """

    X = None
    batches_size = 1024

    for batch in tqdm.tqdm(tensor_batch_generator(data, ft_model, MAX_SEQ_LEN, EMB_SIZE, batches_size),
                           total=len(data) // batches_size + 1):
        if X is None:
            X = batch
            continue
        X = np.vstack([X, batch])

    return X


# if __name__ == '__main__':
#     stop_words = set(nltk.corpus.stopwords.words('russian'))
#     morph = pymorphy2.MorphAnalyzer()
#
#     ft_model_path = CFG.ft_model_path
#     MAX_SEQ_LEN = CFG.MAX_SEQ_LEN
#     EMB_SIZE = CFG.EMB_SIZE
#     sentiment_dict = CFG.sentiment_dict
#     ft_model = FastText.load(ft_model_path)
#
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument("--is_dataframe", help="is your data type dataframe(1=yes, 0=I will use text string)", type=int)
#     parser.add_argument("--data_path_or_text",
#                         help="write path to dataframe_path/text down please, df should has 2 columns-text, sentiment",
#                         type=str, required=True)
#
#     args = parser.parse_args()
#
#     if args.is_dataframe:
#         data = pd.read_csv(args.data_path_or_text)
#         data['tokenized'] = data.iloc[:, 0].apply(tokenize_text)
#         data = data[['tokenized', 'sentiment']].copy()
#         y = data['sentiment'].map(sentiment_dict)
#         X = create_prepared_data(data['tokenized'], ft_model, MAX_SEQ_LEN)
#
#         # разделение данных
#         sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#         train_index, test_index = next(sss.split(X, y))
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#
#         np.save('X_train.npy', X_train)
#         np.save('X_test.npy', X_test)
#         np.save('y_train.npy', y_train)
#         np.save('y_test.npy', y_test)
#
#         print('TRAIN DATA SHAPE: ', X_train.shape, y_train.shape)
#         print('VALIDATION DATA SHAPE: ', X_test.shape, y_test.shape)
#     else:
#         text = args.data_path_or_text
#
#         text_tokenized = tokenize_text(text)
#         text_tokenized_series = pd.Series([text_tokenized])
#         text_prepared = create_prepared_data(text_tokenized_series, ft_model, MAX_SEQ_LEN)
#         print(text_prepared.shape)
#
#         np.save('text.npy', text_prepared)






