from flask import Flask, render_template, url_for, request
from tensorflow import keras
from config import CFG
from preprocessing import *
import numpy as np

import nltk
import pymorphy2
from gensim.models.fasttext import FastText

stop_words = set(nltk.corpus.stopwords.words('russian'))
morph = pymorphy2.MorphAnalyzer()
ft_model_path = CFG.ft_model_path
ft_model = FastText.load(ft_model_path)
MAX_SEQ_LEN = CFG.MAX_SEQ_LEN
EMB_SIZE = CFG.EMB_SIZE


def text_preprocessing(input_text):
    text_prep = tokenize_text(input_text, stop_words, morph)
    text_prep = pd.Series([text_prep])
    text_prep = create_prepared_data(text_prep, ft_model, MAX_SEQ_LEN, EMB_SIZE)
    return text_prep


reversed_sentiment_dict = CFG.reversed_sentiment_dict
saved_model_path_to_dir = CFG.saved_model_path_to_dir

model = keras.models.load_model(saved_model_path_to_dir)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    global my_predictions
    if request.method == 'POST':
        # Класс! Лучший товар! Рекомендую!
        message = request.form['message']
        data = [message]
        print(data[0])
        text = text_preprocessing(data[0])
        my_predictions = model(text)
        print(my_predictions)

    return render_template('result.html', prediction=np.argmax(my_predictions))


if __name__ == '__main__':
    app.run(port=4000)
