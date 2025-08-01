import numpy as np
import pandas as pd
import re
import string
from time import time
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics import F1Score

# импортируем трансформеры
import transformers
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# загрузка токенайзера и  модели, перевод модели в режим предсказания
script_dir = Path(__file__).parent.resolve()
PATH = script_dir.parent / 'models' / 'model_rtt.pth'
model_full = torch.load(PATH)
model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
tokenizer_rtt = AutoTokenizer.from_pretrained(model_checkpoint)
model_full.eval()

# предсказание на основе полученного текста
def text2toxicity(text, model=model_full):
    probably_box = []
    with torch.no_grad():
        # Токенизируем текст и перемещаем на нужное устройство
        inputs = tokenizer_rtt(text, return_tensors='pt', truncation=True, padding=True).to(device)
        # Токенизируем текст и перемещаем на нужное устройство
        proba = torch.sigmoid(model(**inputs).logits).squeeze(1).cpu().numpy()
    for p in proba:
        if p > 0.5:
            probably_box.append(f'Оскорбительный, p={p:.2f}')
        else:
            probably_box.append(f'Не оскорбительный, p={p:.2f}')
    return pd.DataFrame({'Исходный текст': text, 
                         'Прогноз': probably_box})


st.title("🦜Приложение для определения токсичности комментариев чатов")
st.header('Расчет вероятности токсичности текста')

txt_box = st.text_input("Вставьте текст в рамку для расчета вероятности")
if txt_box:
    st.write(text2toxicity(txt_box))

uploaded_file = st.file_uploader("Выберите файл в формате *.txt", type='txt')
if uploaded_file is not None:
    lw = pd.read_csv(uploaded_file, header=None)[0].tolist()
    st.write(text2toxicity(lw))

st.header('Основные данные о модели')
st.write('**Задача:** обучить модель способную распозновать комментарии из социальных сетей оскорбительного или токсичного характера')
st.write('**Название базовой модели:** rubert-tiny-toxicity')
st.write('**Размер модели:** 11.8M параметров')
st.write('**Размер датасета для обучения**: 14412 строк')
st.write('**О датасете**: содержит размеченные комментарии из российской социальной сети ok.ru')
st.write('**Лучший показатель loss train / valid**: 0.271 / 0.280')
st.write('**Лучший показатель accuracy train / valid**: 0.898 / 0.888')
st.write('**Лучший показатель f1_score**: 0.842')


st.header('График метрик модели во время обучения')

PATH_M= script_dir.parent / 'models' / 'rtt_metrics.json'
df_metrics = pd.read_json(PATH_M)
st.line_chart(df_metrics, x='epochs', x_label='эпохи', height=600)

st.header('Сравнение предсказанных и истинных признаков')
PATH_I= script_dir.parent / 'images' / 'c_matrix_rtt.png'
st.image(PATH_I, caption=None)
