import sklearn.model_selection
import tensorflow as tf
from tensorflow import keras
# from tensorflow.python.keras.layers import GRU, Embedding, Dense, Input, Dropout, CuDNNGRU, LSTM
from keras.layers import GRU, Embedding, Dense, Input, Dropout, CuDNNGRU, LSTM, Bidirectional
from tensorflow.python.keras.models import Sequential
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import pad_sequences
import re
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import csv
from pyarabic.araby import strip_tatweel
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
nltk.download('stopwords')
stop_words = list(set(stopwords.words('arabic')))

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations


def preprocess(text):
    out = re.sub(r"(\r\n|\n|\r)", '', text)
    out = re.sub(r"[^\w\s]", ' ', out)
    out = re.sub(r"[a-zA-Z]", '', out)
    out = re.sub(r"\n", '', out)
    out = re.sub(r"\s+", ' ', out)
    out = re.sub(r"[\_]", ' ', out)
    out = strip_tatweel(out)
    out = re.sub(r"(.)\1{2,}", r'\1', out)
    out = re.sub(r"[0-9]", '', out)
    out = re.sub(r"(٠|١|٢|٣|٤|٥|٦|٧|٨|٩)", '', out)
    out = remove_emoji(out)
    out = remove_diacritics(out)
    stemmer = nltk.ISRIStemmer()
    stemmer.stem(out)
    tokens = word_tokenize(out)
    out = ' '.join([word for word in tokens if word not in stop_words])

    return out.strip()


def remove_emoji(text):
    regrex_pattern = re.compile(pattern="["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'',text)


def remove_diacritics(text):
    arabic_diacritics = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    text = re.sub(arabic_diacritics, '', str(text))
    return text

def make_prediction(message):
    print(model.predict(message))

# def process_review(review):
#     out = re.sub(r"[^\w\s]", '', review)
#     out = re.sub(r"[a-zA-Z]", '', out)
#     out = re.sub(r"\n", '', out)
#     out = re.sub(r"\s+", ' ', out)
#     out = strip_tatweel(out)
#     return out.strip()


# def classify(sentence):
#     class_names = ['سلبي', 'إيجابي']
#     sentence = preprocess(sentence)
#     sequence = [tokenizer.word_index[word] for word in sentence.split(' ')]
#     sequence = pad_sequences([sequence], maxlen=X.shape[1], padding='post', value=0)
#     pred = model.predict(sequence)[0][0]
#     print(class_names[np.round(pred).astype('int')], pred)


# def classify(sentence):
#     class_names = ['سلبي', 'إيجابي']
#     sequence = []
#     sentence = process_review(sentence)
#     for word in sentence.split(' '):
#         if tknzr.word_index.get(word):
#             sequence.append(tknzr.word_index[word])
#         else:
#             for letter in list(word):
#                 if tknzr.word_index.get(letter):
#                     sequence.append(tknzr.word_index[letter])
#                 else:
#                     continue
#     sequence = pad_sequences([sequence], maxlen=X.shape[1], padding='post', value=0)
#     pred = model.predict(sequence)[0][0]
#     print(class_names[np.round(pred).astype('int')], pred)


if __name__ == '__main__':
    pd.set_option('display.max_colwidth', None)

    df = pd.read_csv('Arabic_Spam.csv')

    train_df, test_df, train_labels, test_labels = sklearn.model_selection.train_test_split(df[df.columns[0]], df[df.columns[1]], test_size=0.25)

    train_df = train_df.apply(preprocess)
    test_df = test_df.apply(preprocess)

    print(train_df.shape)

    vec = CountVectorizer()
    train_df = vec.fit_transform(train_df).toarray()
    test_df = vec.transform(test_df).toarray()

    model = MultinomialNB()
    model.fit(train_df, train_labels)

    print(model.score(test_df, test_labels))


    # # train_df = train_df.drop(columns=["Column1"])
    # # train_df = train_df.drop(columns=["Column2"])
    # # test_df = test_df.drop(columns=["Column1"])
    # test_df = test_df.drop(columns=["Column2"])
    #
    # labels_train = train_df['Sentiment']
    # labels_test = test_df['Sentiment']
    #
    # del train_df['Sentiment']
    # del test_df['Sentiment']
    #
    # tweets_train = train_df['Tweet']
    # tweets_test = test_df['Tweet']
    #
    # tokenizer = Tokenizer(lower=True, split=' ', oov_token=True)
    # tokenizer.fit_on_texts(tweets_train)
    # X = tokenizer.texts_to_sequences(tweets_train)
    # X = pad_sequences(X, padding='post', value=0)
    #
    # X = np.array(X)
    # labels_train = np.array(labels_train)
    # print(X.shape)
    #
    # model = Sequential()
    # model.add(Embedding(len(tokenizer.word_index)+1, 32))
    # model.add(Bidirectional(GRU(units=32)))
    # model.add(Dense(32, activation='tanh'))
    # model.add(Dropout(0.3))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.build((4272, 239))
    # model.summary()
    #
    # model.fit(X, labels_train, validation_split=0.1, epochs=7, batch_size=128, shuffle=True)
    #
    # classify("جميل")

    # with open('Arabic Spam Dataset.csv', 'r') as csv_file:
    #     reviews = []
    #     labels = []
    #     all_text = ""
    #     count = 0
    #     pos_count = 0
    #
    #     # read the data
    #     lines = csv.reader(csv_file, delimiter=",")
    #     for i, line in enumerate(lines):
    #
    #         # preprocess the data
    #         review = process_review(line[0])
    #         label = int(line[1])
    #
    #         # only allow postiive and negative reviews,
    #         # also make them the same length
    #
    #         if label == 1:
    #             pos_count += 1
    #         elif label == 0:
    #             continue
    #         else:
    #             label += 1
    #
    #         if label == 1 and pos_count > 862:
    #             continue
    #
    #         if review == "":
    #             continue
    #         reviews.append(review)
    #         all_text += review + ' \n '
    #         labels.append(label)
    #         print(reviews)
    #
    # # shuffle the data
    # reviews, labels = shuffle(reviews, labels)
    #
    # tknzr = Tokenizer(lower=True, split=" ", oov_token='OOV', num_words=None)
    # tknzr.fit_on_texts(reviews)
    #
    # # making sequences:
    # X = tknzr.texts_to_sequences(reviews)
    # X = pad_sequences(X, padding='post', value=0)
    #
    # X = np.array(X)
    # y = np.array(labels)
    #
    # model = Sequential()
    # model.add(Embedding(len(tknzr.word_index)+1, 32))
    # model.add(Bidirectional(GRU(units=32)))
    # model.add(Dropout(0.3))
    # model.add(Dense(32, activation='tanh'))
    # model.add(Dropout(0.3))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #
    # es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    #
    # model.fit(X, y, validation_split=0.1, epochs=7, batch_size=128, shuffle=True, callbacks=[es_callback])
    #
    # classify('اتصل للتحديث')






