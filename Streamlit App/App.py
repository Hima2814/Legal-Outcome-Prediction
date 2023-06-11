from transformers import TFBertModel, TFBertForSequenceClassification, DistilBertTokenizer, TFDistilBertModel, RobertaTokenizer, TFRobertaModel
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from keras_preprocessing.sequence import pad_sequences
import streamlit as st
from PIL import Image
import tensorflow.keras as keras
import base64


import streamlit as st


class CustomBertModel(TFBertModel):
    pass


stop = list(stopwords.words('english'))
new_stop = ['(', ',', '.', '..', '"', ')', '>', '’', '<', '“', '/', '?']
stop.extend(new_stop)

porter = PorterStemmer()


def remove_punctuation(description):
    """Function to remove punctuation"""
    table = str.maketrans('', '', string.punctuation)
    return description.translate(table)


def remove_stopwords(text):
    """Function to removing stopwords"""
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)


def stemmer(stem_text):
    """Function to apply stemming"""
    stem_text = [porter.stem(word) for word in stem_text.split()]
    return " ".join(stem_text)


label_to_class = {0: 'Appeal Allowed', 1: 'Appeal Dismissed',
                  2: 'Petition Dismissed', 3: 'Petition Allowed'}

# Title
st.header(':judge: :scales: Legal Outcome Prediction :scales: :female-judge:')


sidebar = st.sidebar
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        width: 200px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.title(
    ":question: Previews of the Models :question:")
st.sidebar.subheader(
    "View the architecture, the train and test accuracy and the learning curves of the models.")

# Entering the case file
CaseFile = st.text_area("Enter the case file below", height=100)

# Model Previews
if sidebar.button("-> Select Model <-"):
    st.sidebar.write(
        "Click on the buttons below to get the visualizations of the architecture, the train and test accuracy and the learning curve of that model.")
# MLP Model
if sidebar.button("MLP Model"):
    image = Image.open("MLP_Architecture.png")
    sidebar.image(image, use_column_width=True)

# CNN Model
if sidebar.button("CNN Model"):
    image = Image.open("CNN_Architecture.png")
    sidebar.image(image, use_column_width=True)

# Text-CNN Model
if sidebar.button("Text-CNN Model"):
    image = Image.open("Text-CNN_Architecture.png")
    sidebar.image(image, use_column_width=True)

# BERT Model
if sidebar.button("BERT Model"):
    image = Image.open("BERT_Architecture.png")
    sidebar.image(image, use_column_width=True)

# LegalBERT Model
if sidebar.button("LegalBERT Model"):
    image = Image.open("LegalBERT_Architecture.png")
    sidebar.image(image, use_column_width=True)

# DistilBERT Model
if sidebar.button("DistilBERT Model"):
    image = Image.open("DistilBERT_Architecture.png")
    sidebar.image(image, use_column_width=True)

# RoBERTa Model
if sidebar.button("RoBERTa Model"):
    image = Image.open("RoBERTa_Architecture.png")
    sidebar.image(image, use_column_width=True)

# BERT+LegalBERT Model
if sidebar.button("BERT + LegalBERT Model"):
    image = Image.open("BERT+LegalBERT_Architecture.png")
    sidebar.image(image, use_column_width=True)

# BERT+DistilBERT Model
if sidebar.button("BERT + DistilBERT Model"):
    image = Image.open("BERT+DistilBERT_Architecture.png")
    sidebar.image(image, use_column_width=True)

# BERT+RoBERTa Model
if sidebar.button("BERT + RoBERTa Model"):
    image = Image.open("BERT+RoBERTa_Architecture.png")
    sidebar.image(image, use_column_width=True)

# DistilBERT+LegalBERT Model
if sidebar.button("DistilBERT + LegalBERT Model"):
    image = Image.open("DistilBERT+LegalBERT_Architecture.png")
    sidebar.image(image, use_column_width=True)

# DistilBERT+BERT Model
if sidebar.button("DistilBERT + BERT Model"):
    image = Image.open("DistilBERT+BERT_Architecture.png")
    sidebar.image(image, use_column_width=True)

# DistilBERT+RoBERTa Model
if sidebar.button("DistilBERT + RoBERTa Model"):
    image = Image.open("DistilBERT+RoBERTa_Architecture.png")
    sidebar.image(image, use_column_width=True)

Model = st.selectbox(
    "Model Selection", ["Select a model", "MLP", "CNN", "Text_CNN", "BERT", "LegalBERT", "DistilBERT", "RoBERTa", "BERT + LegalBERT", "BERT + DistilBERT", "BERT + RoBERTa",  "DistilBERT + BERT", "DistilBERT + LegalBERT", "DistilBERT + RoBERTa"])

if CaseFile:
    if Model == 'MLP':
        tokenize = Tokenizer(num_words=10000)
        CaseFile = remove_punctuation(CaseFile)
        CaseFile = remove_stopwords(CaseFile)
        CaseFile = stemmer(CaseFile)
        input_sequence = tokenize.texts_to_sequences([CaseFile])
        input_sequence = pad_sequences(
            input_sequence, padding='post', maxlen=1000)
        model_2 = tf.keras.models.load_model("MLP.h5")
        predicted_labels = model_2.predict(input_sequence)
        predicted_label = np.argmax(predicted_labels)
        st.text(
            f"The predicted judgement status of the given case file is: {label_to_class[predicted_label]}")

    if Model == 'CNN':
        tokenize = Tokenizer(num_words=10000)
        CaseFile = remove_punctuation(CaseFile)
        CaseFile = remove_stopwords(CaseFile)
        CaseFile = stemmer(CaseFile)
        input_sequence = tokenize.texts_to_sequences([CaseFile])
        input_sequence = pad_sequences(
            input_sequence, padding='post', maxlen=1000)
        model_2 = tf.keras.models.load_model('CNN.h5')
        predicted_labels = model_2.predict(input_sequence)
        predicted_label = np.argmax(predicted_labels)
        st.text(
            f"The predicted judgement status of the given case file is: {label_to_class[predicted_label]}")

    if Model == 'Text_CNN':
        tokenize = Tokenizer(num_words=10000)
        CaseFile = remove_punctuation(CaseFile)
        CaseFile = remove_stopwords(CaseFile)
        CaseFile = stemmer(CaseFile)
        input_sequence = tokenize.texts_to_sequences([CaseFile])
        input_sequence = pad_sequences(
            input_sequence, padding='post', maxlen=1000)
        model_2 = tf.keras.models.load_model('TextCNN.h5')
        predicted_labels = model_2.predict(input_sequence)
        predicted_label = np.argmax(predicted_labels)
        st.text(
            f"The predicted judgement status of the given case file is: {label_to_class[predicted_label]}")

    if Model == 'BERT':
        # Register the custom layer class with Keras
        tf.keras.utils.get_custom_objects()['TFBertModel'] = CustomBertModel
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        CaseFile = remove_punctuation(CaseFile)
        CaseFile = remove_stopwords(CaseFile)
        CaseFile = stemmer(CaseFile)
        custom_input = tokenizer(text=CaseFile,
                                 add_special_tokens=True,
                                 max_length=100,
                                 truncation=True,
                                 padding=True,
                                 return_tensors='tf',
                                 return_token_type_ids=False,
                                 return_attention_mask=True
                                 )
        model_2 = tf.keras.models.load_model('BERT.h5')
        predictions = model_2.predict(
            {'input_ids': custom_input['input_ids'], 'attention_mask': custom_input['attention_mask']})
        predicted_label = np.argmax(predictions, axis=1)[0]
        st.text(
            f"The predicted judgement status of the given case file is: {label_to_class[predicted_label]}")

    if Model == 'LegalBERT':
        tokenizer = AutoTokenizer.from_pretrained(
            "nlpaueb/legal-bert-base-uncased")
        CaseFile = remove_punctuation(CaseFile)
        CaseFile = remove_stopwords(CaseFile)
        CaseFile = stemmer(CaseFile)
        custom_input = tokenizer(
            text=CaseFile,
            add_special_tokens=True,
            max_length=100,
            truncation=True,
            padding=True,
            return_tensors='tf',
            return_token_type_ids=False,
            return_attention_mask=True
        )
        custom_objects = {
            'TFBertForSequenceClassification': TFBertForSequenceClassification
        }
        with keras.utils.custom_object_scope(custom_objects):
            model_2 = tf.keras.models.load_model("LegalBERT.h5")
        predictions = model_2.predict(
            {'input_ids': custom_input['input_ids'], 'attention_mask': custom_input['attention_mask']})
        predicted_label = np.argmax(predictions, axis=1)[0]
        st.text(
            f"The predicted judgement status of the given case file is: {label_to_class[predicted_label]}")

    if Model == 'DistilBERT':
        dbert_tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased')
        CaseFile = remove_punctuation(CaseFile)
        CaseFile = remove_stopwords(CaseFile)
        CaseFile = stemmer(CaseFile)
        custom_input_tokens = dbert_tokenizer.encode_plus(
            CaseFile, add_special_tokens=True, max_length=100, pad_to_max_length=True, return_attention_mask=True, truncation=True)
        custom_input_ids = np.asarray(custom_input_tokens['input_ids'])
        custom_attention_mask = np.asarray(
            custom_input_tokens['attention_mask'])

        custom_input_ids = custom_input_ids.reshape((1, -1))
        custom_attention_mask = custom_attention_mask.reshape((1, -1))
        custom_objects = {
            'TFDistilBertModel': TFDistilBertModel,
            'TFBertForSequenceClassification': TFBertForSequenceClassification
        }
        with keras.utils.custom_object_scope(custom_objects):
            model_2 = tf.keras.models.load_model("DistilBERT.h5")
        predictions = model_2.predict(
            [custom_input_ids, custom_attention_mask])
        predicted_label = np.argmax(predictions[0], axis=-1)
        st.text(
            f"The predicted judgement status of the given case file is: {label_to_class[predicted_label]}")

    if Model == 'RoBERTa':
        roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        CaseFile = remove_punctuation(CaseFile)
        CaseFile = remove_stopwords(CaseFile)
        CaseFile = stemmer(CaseFile)
        roberta_inp = roberta_tokenizer.encode_plus(
            CaseFile, add_special_tokens=True, max_length=100, pad_to_max_length=True, truncation=True)
        input_ids = np.asarray(roberta_inp['input_ids']).reshape(1, -1)
        attention_mask = np.asarray(
            roberta_inp['attention_mask']).reshape(1, -1)
        custom_objects = {
            'TFRobertaModel': TFRobertaModel  # ,
            # 'TFBertForSequenceClassification': TFBertForSequenceClassification
        }
        with keras.utils.custom_object_scope(custom_objects):
            model_2 = tf.keras.models.load_model("RoBERTa.h5")
        predictions = model_2.predict([input_ids, attention_mask])
        predicted_label = np.argmax(predictions, axis=1)[0]
        st.text(
            f"The predicted judgement status of the given case file is: {label_to_class[predicted_label]}")

    if Model == 'BERT + DistilBERT':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        CaseFile = remove_punctuation(CaseFile)
        CaseFile = remove_stopwords(CaseFile)
        CaseFile = stemmer(CaseFile)
        custom_input_tokens = tokenizer.encode_plus(
            CaseFile, add_special_tokens=True, max_length=100, pad_to_max_length=True, return_attention_mask=True, truncation=True)
        custom_input_ids = np.asarray(custom_input_tokens['input_ids'])
        custom_attention_mask = np.asarray(
            custom_input_tokens['attention_mask'])

        custom_input_ids = custom_input_ids.reshape((1, -1))
        custom_attention_mask = custom_attention_mask.reshape((1, -1))
        custom_objects = {
            'TFDistilBertModel': TFDistilBertModel,
            'TFBertForSequenceClassification': TFBertForSequenceClassification
        }
        with keras.utils.custom_object_scope(custom_objects):
            model_2 = tf.keras.models.load_model("BERT-DistilBert.h5")
        predictions = model_2.predict(
            [custom_input_ids, custom_attention_mask])
        predicted_label = np.argmax(predictions[0], axis=-1)
        st.text(
            f"The predicted judgement status of the given case file is: {label_to_class[predicted_label]}")

    if Model == 'BERT + RoBERTa':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        CaseFile = remove_punctuation(CaseFile)
        CaseFile = remove_stopwords(CaseFile)
        CaseFile = stemmer(CaseFile)
        roberta_inp = tokenizer.encode_plus(
            CaseFile, add_special_tokens=True, max_length=100, pad_to_max_length=True, truncation=True)
        input_ids = np.asarray(roberta_inp['input_ids']).reshape(1, -1)
        attention_mask = np.asarray(
            roberta_inp['attention_mask']).reshape(1, -1)
        custom_objects = {
            'TFRobertaModel': TFRobertaModel  # ,
            # 'TFBertForSequenceClassification': TFBertForSequenceClassification
        }
        with keras.utils.custom_object_scope(custom_objects):
            model_2 = tf.keras.models.load_model("BERT-Roberta.h5")
        predictions = model_2.predict([input_ids, attention_mask])
        predicted_label = np.argmax(predictions, axis=1)[0]
        st.text(
            f"The predicted judgement status of the given case file is: {label_to_class[predicted_label]}")

    if Model == 'BERT + LegalBERT':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        CaseFile = remove_punctuation(CaseFile)
        CaseFile = remove_stopwords(CaseFile)
        CaseFile = stemmer(CaseFile)
        custom_input = tokenizer(
            text=CaseFile,
            add_special_tokens=True,
            max_length=100,
            truncation=True,
            padding=True,
            return_tensors='tf',
            return_token_type_ids=False,
            return_attention_mask=True
        )
        custom_objects = {
            'TFBertForSequenceClassification': TFBertForSequenceClassification
        }
        with keras.utils.custom_object_scope(custom_objects):
            model_2 = tf.keras.models.load_model("BERT-LegalBert.h5")
        predictions = model_2.predict(
            {'input_ids': custom_input['input_ids'], 'attention_mask': custom_input['attention_mask']})
        predicted_label = np.argmax(predictions, axis=1)[0]
        st.text(
            f"The predicted judgement status of the given case file is: {label_to_class[predicted_label]}")

    if Model == 'DistilBERT + BERT':
        dbert_tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased')
        tf.keras.utils.get_custom_objects()['TFBertModel'] = CustomBertModel
        CaseFile = remove_punctuation(CaseFile)
        CaseFile = remove_stopwords(CaseFile)
        CaseFile = stemmer(CaseFile)
        custom_input_tokens = dbert_tokenizer.encode_plus(
            CaseFile, add_special_tokens=True, max_length=100, pad_to_max_length=True, return_attention_mask=True, truncation=True)
        custom_input_ids = np.asarray(custom_input_tokens['input_ids'])
        custom_attention_mask = np.asarray(
            custom_input_tokens['attention_mask'])

        custom_input_ids = custom_input_ids.reshape((1, -1))
        custom_attention_mask = custom_attention_mask.reshape((1, -1))

        # Load the trained model
        model_2 = tf.keras.models.load_model('DistilBERT-Bert.h5')

        # Make predictions
        predictions = model_2.predict(
            {'input_1': custom_input_ids, 'input_2': custom_attention_mask})

        predicted_label = np.argmax(predictions, axis=1)[0]

        st.text(
            f"The predicted judgement status of the given case file is: {label_to_class[predicted_label]}")

if Model == 'DistilBERT + LegalBERT':
    dbert_tokenizer = DistilBertTokenizer.from_pretrained(
        'distilbert-base-uncased')
    tf.keras.utils.get_custom_objects()['TFBertModel'] = CustomBertModel
    CaseFile = remove_punctuation(CaseFile)
    CaseFile = remove_stopwords(CaseFile)
    CaseFile = stemmer(CaseFile)
    custom_input_tokens = dbert_tokenizer.encode_plus(
        CaseFile, add_special_tokens=True, max_length=100, pad_to_max_length=True, return_attention_mask=True, truncation=True)
    custom_input_ids = np.asarray(custom_input_tokens['input_ids'])
    custom_attention_mask = np.asarray(custom_input_tokens['attention_mask'])

    custom_input_ids = custom_input_ids.reshape((1, -1))
    custom_attention_mask = custom_attention_mask.reshape((1, -1))
    custom_objects = {
        'TFBertForSequenceClassification': TFBertForSequenceClassification
    }
    with keras.utils.custom_object_scope(custom_objects):
        model_2 = tf.keras.models.load_model('DistilBERT-LegalBERT.h5')
    predictions = model_2.predict([custom_input_ids, custom_attention_mask])
    predicted_label = np.argmax(predictions, axis=1)[0]

    st.text(
        f"The predicted judgement status of the given case file is: {label_to_class[predicted_label]}")

if Model == 'DistilBERT + RoBERTa':
    dbert_tokenizer = DistilBertTokenizer.from_pretrained(
        'distilbert-base-uncased')
    tf.keras.utils.get_custom_objects()['TFBertModel'] = CustomBertModel
    CaseFile = remove_punctuation(CaseFile)
    CaseFile = remove_stopwords(CaseFile)
    CaseFile = stemmer(CaseFile)
    custom_input_tokens = dbert_tokenizer.encode_plus(
        CaseFile, add_special_tokens=True, max_length=100, pad_to_max_length=True, return_attention_mask=True, truncation=True)
    custom_input_ids = np.asarray(custom_input_tokens['input_ids'])
    custom_attention_mask = np.asarray(custom_input_tokens['attention_mask'])

    custom_input_ids = custom_input_ids.reshape((1, -1))
    custom_attention_mask = custom_attention_mask.reshape((1, -1))
    custom_objects = {
        'TFRobertaModel': TFRobertaModel  # ,
        # 'TFBertForSequenceClassification': TFBertForSequenceClassification
    }
    with keras.utils.custom_object_scope(custom_objects):
        model_2 = tf.keras.models.load_model('DistilBERT-RoBERTa.h5')
    predictions = model_2.predict([custom_input_ids, custom_attention_mask])
    predicted_label = np.argmax(predictions, axis=1)[0]

    st.text(
        f"The predicted judgement status of the given case file is: {label_to_class[predicted_label]}")
