import numpy as np
import sys
import json
import requests
import nltk
import pymorphy2
from gensim.models.fasttext import FastText
import tqdm
import pandas as pd
import string
import logging
import contextlib
from http.client import HTTPConnection


def debug_requests_on():
    """Switches on logging of the requests module."""
    HTTPConnection.debuglevel = 1

    logging.basicConfig(filename='logs.log')
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True


def debug_requests_off():
    """Switches off logging of the requests module, might be some side-effects"""
    HTTPConnection.debuglevel = 0

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    root_logger.handlers = []
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.WARNING)
    requests_log.propagate = False


@contextlib.contextmanager
def debug_requests():
    """ Use with 'with'! """
    debug_requests_on()
    yield
    debug_requests_off()


def tokenize_text(raw_text: str):
    """Токенизирует текст, который подается на вход"""
    tokenized_str = nltk.word_tokenize(raw_text)
    tokens = [i.lower() for i in tokenized_str if (i not in string.punctuation + '...' + '``' + "''" + "»" + '–' + '—')]
    filtered_tokens = [i for i in tokens if (i not in stop_words)]
    lemmatized_tokens = [morph.parse(i)[0].normal_form for i in filtered_tokens]
    return lemmatized_tokens


stop_words = set(nltk.corpus.stopwords.words('russian'))  # стоп слова
morph = pymorphy2.MorphAnalyzer()  # для лемматизации

print('loading FastText model...')
ft_model = FastText.load("ft_model.model")  # загрузка предобученной модели fasttext
print('complete!')

MAX_SEQ_LEN = 512  # максимальная длина последовательности
EMB_SIZE = 32  # размер векторного представления
sentiment_dict = {'neutral': 1, 'neautral': 1, 'negative': 0, 'positive': 2}
reversed_sentiment_dict = {index: gen for gen, index in sentiment_dict.items()}  # для понятного предсказания модели


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


def padding_sequence(list_of_vectors, MAX_SEQ_LEN):
    """
    преобразовать список векторов в массив формы (MAX_SEQ_LEN, EMB_SIZE), добавив 0 к коротким предложениям (дополнение сообщения)
    или обрезка до MAX_SEQ_LEN для длинных предложений
    к концу, размер вывода функции равен (1, MAX_SEQ_LEN, EMB_SIZE)
    """

    sequence = np.array(list_of_vectors)
    if MAX_SEQ_LEN > sequence.shape[0] > 0:
        sequence = np.vstack([sequence, np.zeros(shape=(MAX_SEQ_LEN - sequence.shape[0], sequence.shape[1]))])
    elif sequence.shape[0] > MAX_SEQ_LEN:
        sequence = sequence[:MAX_SEQ_LEN]
    elif sequence.shape[0] == 0:
        sequence = np.ones((MAX_SEQ_LEN, EMB_SIZE))
    assert sequence.shape == (MAX_SEQ_LEN,
                              EMB_SIZE),\
        f'padding_sequence output is incorrect: {sequence.shape} instead of {(MAX_SEQ_LEN, EMB_SIZE)}'

    return sequence[np.newaxis, ...].astype(np.float32)


def tensor_batch_generator(data, ft_model, MAX_SEQ_LEN, batch_size=1024):
    """
    Генератор используется для экономии памяти во время векторизации наборов данных трейна / тест.
    data - pd.Series (или другой итеративный тип) со списками токенов, например: df ['tokenized']
    """
    length = len(data)
    for i in range(0, length, batch_size):
        X = []
        for tokens in data[i:min(i + batch_size, length)]:
            X.append(padding_sequence(encode_to_vector(tokens, ft_model), MAX_SEQ_LEN))
        yield np.vstack(X)


def create_prepared_data(data, ft_model, MAX_SEQ_LEN, batches_size=1014):
    """
    Создать массив векторизованных последовательностей с правым заполнением из токенизированного текста
    """
    X = None
    batches_size = 1024

    for batch in tqdm.tqdm(tensor_batch_generator(data, ft_model, MAX_SEQ_LEN, batches_size),
                           total=len(data) // batches_size + 1):
        if X is None:
            X = batch
            continue
        X = np.vstack([X, batch])

    return X


print('preparing your text...')
input_text = sys.argv[1:]  # чтение декста из консоли(все, что будет введено в консоли после вызова файла, будет здесь)
input_text = ' '.join(input_text)  # из списка, который мы получили, нужно сделать строку

input_text_tokenized = tokenize_text(input_text)  # токенизация текста
input_text_tokenized_series = pd.Series([input_text_tokenized])  # привидение к нужному формату для корректнгой работы
text_prepared = create_prepared_data(input_text_tokenized_series, ft_model, MAX_SEQ_LEN)  # векторизация, привидение к нужному размеру
print('complete!')

with debug_requests():  #
    print('connecting to the server...')
    # Подготовка данных для HTTP запроса
    request_data = json.dumps({
        "signature_name": "serving_default",
        "instances": text_prepared.tolist()
    })
    headers = {"content-type": "application/json"}

    # HTTP запрос на сервер
    json_response = requests.post(
        'http://localhost:8601/v1/models/senti_model:predict',
        data=request_data, headers=headers)
    print('complete!')

print('getting predictions...')
# Обработка JSON ответа и получение предсказаний
predictions = json.loads(json_response.text)['predictions']
prediction = reversed_sentiment_dict[np.argmax(predictions)]
print('prediction is', prediction)

json_resp = json.loads(json_response.text)
print(json_resp)
