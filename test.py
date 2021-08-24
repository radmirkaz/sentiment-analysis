import tensorflow as tf
from config import CFG


reversed_sentiment_dict = CFG.reversed_sentiment_dict
saved_model_path_to_dir = CFG.saved_model_path_to_dir

model = tf.keras.models.load_model(saved_model_path_to_dir)

print(model('model'))