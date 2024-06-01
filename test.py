import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import json
import tensorflow_text
import keras_nlp
from sklearn.model_selection import train_test_split

data_json_path = '/Users/yangchengchang/PycharmProjects/BERT/bert/DATA/sa_data/NewsHeadlines1/Sarcasm_Headlines_Dataset.json'


data = []

# pen the file and read line by line
with open(data_json_path, 'r') as file:
    for line in file:
        # parse each line as a JSON object and append to the list
        data.append(json.loads(line))

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)
df.drop_duplicates(inplace=True)

# Split the dataset into train and test sets
X_temp, X_test, y_temp, y_test = train_test_split(df["headline"],
                                                    df["is_sarcastic"],
                                                    test_size=0.30,
                                                    random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp,
                                                  y_temp,
                                                  test_size=0.20,
                                                  random_state=42)

# We choose 512 because it's the limit of DistilBert
SEQ_LENGTH = 64

# Use a shorter sequence length.
bert_preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
    "bert_large_en_uncased",
    sequence_length=SEQ_LENGTH,
)

# Pretrained classifier.
model = keras_nlp.models.BertClassifier.from_preset(
    "bert_large_en_uncased",
    num_classes=2,
    activation=None,
    preprocessor=bert_preprocessor,
)
# intialize an instance
# model = BertClassifier(bert_preprocessor, bert_encoder, seq_length=256)


model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.legacy.Adam(1e-5),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy()
   ]
)

from tensorflow.keras.callbacks import ModelCheckpoint

epochs = 2

metric ='val_auc'
callback_list = [
                 tf.keras.callbacks.EarlyStopping(monitor=metric,
                                                  patience= 1,
                                                  restore_best_weights=True),
                 ModelCheckpoint('model_best.keras', monitor='val_loss', save_best_only=True, mode='min')
                ]
history = model.fit(x=X_train,
                    y=y_train,
                    validation_data=(X_test, y_test),
                    epochs=epochs,
                    batch_size=32,
                    callbacks=[callback_list])


model.save('model_best_bert.keras')  # 保存整个模型到 HDF5 文件