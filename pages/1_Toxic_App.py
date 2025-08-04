import numpy as np
import pandas as pd
import re
import string
from time import time
from functools import partial
from pathlib import Path
import shutil
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics import F1Score

# импортируем трансформеры
import transformers
from transformers import AutoModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from huggingface_hub import login, whoami, model_info

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# загрузка токенайзера и модели, перевод модели в режим предсказания
# Укажите имя модели с Hugging Face
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "pampad/mymodel_ruber_tt_1label" 
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

model_full, tokenizer_rtt = load_model_and_tokenizer()
model_full.eval()

def data_preprocessing(text: str) -> str:
    text = text.lower()
    text = re.sub(r'(https://\w.*).*|(http://\w.*).*', '', text)
    text = ''.join([c for c in text if c not in string.punctuation]) # Remove punctuation
    # text = ''.join(text)
    return text

# функцция предсказания модели на основе полученного текста
def text2toxicity(text):
    """Определение токсичности текста"""
    try:
        inputs = tokenizer_rtt(
            data_preprocessing(text),
            return_tensors="pt", 
            truncation=True,
            padding=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = model_full(**inputs)
            logits = outputs.logits
            
            # Используем softmax для получения вероятностей
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
            
            # Берем вероятность токсичного класса (обычно индекс 1)
            if len(proba[0]) > 1:
                return float(proba[0][1])  # Токсичный класс
            else:
                return float(proba[0][0])
                
    except Exception as e:
        st.error(f"Ошибка при анализе текста: {str(e)}")
        return 0.0

# **STREAMLIT**

## блок для загрузки исходных данных и вывода предсказания модели
st.title("Приложение для определения токсичности комментариев чатов")
st.header('Расчет вероятности токсичности текста')

txt_box = st.text_input("Вставьте текст в рамку для расчета вероятности")
if txt_box:
    p = text2toxicity(txt_box)
    if p > 0.5:
        st.write(f"Текст токсичный, p={round(p, 3)}")
    else:
        st.write(f"Текст не токсичный, p={round(p, 3)}")

uploaded_file = st.file_uploader("Выберите файл в формате *.txt", type='txt')
if uploaded_file is not None:
    try:
        content = uploaded_file.read().decode('utf-8')
        lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
        lines_clear = [data_preprocessing(line).strip() for line in content.strip().split('\n') if line.strip()]

        if lines:
            st.write(f"Загружено строк для анализа: {len(lines)}")
            
            # Пакетная токенизация
            inputs = tokenizer_rtt(lines_clear, return_tensors="pt", truncation=True, padding=True, max_length=512)

            with torch.no_grad():
                outputs = model_full(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy() # Вероятность токсичного класса

            # Формирование результатов
            results = []
            for i, line in enumerate(lines):
                prob = probabilities[i]
                results.append({
                    'Текст': line[:150] + "..." if len(line) > 150 else line,
                    'Вероятность': round(float(prob), 3),
                    'Токсичный': "Да" if prob > 0.5 else "Нет"
                })

            df_results = pd.DataFrame(results)
            st.dataframe(df_results)

            toxic_count = sum(1 for p in probabilities if p > 0.5)
            st.write(f"**Статистика:** {toxic_count} из {len(lines)} строк токсичны")

    except Exception as e:
        st.error(f"Ошибка при обработке файла: {str(e)}")

## блок с данными о модели и датасете
st.header('Основные данные о модели')
st.write('**Задача:** обучить модель способную распозновать комментарии из социальных сетей оскорбительного или токсичного характера')
st.write('**Название базовой модели:** rubert-tiny-toxicity')
st.write('**Размер модели:** 11.8M параметров')
st.write('**Размер датасета для обучения**: 14412 комментариев на русском языке с разметкой классов')
st.write('**Количество классов датасета для обучения**: 2 класса - токсичный/нетоксичный в пропорции 1:3')
st.write('**О датасете**: содержит размеченные комментарии из российской социальной сети ok.ru')
st.write('**Лучший показатель loss train / valid**: 0.271 / 0.280')
st.write('**Лучший показатель accuracy train / valid**: 0.898 / 0.888')
st.write('**Лучший показатель f1_score**: 0.842')

## блок построения графика метрик
st.header('График метрик модели во время обучения')
script_dir = Path(__file__).parent.parent.absolute()
PATH_M= script_dir / 'models' / 'rtt_metrics.json'          # !!!ПУТЬ к метрикам (поменять на свой)
df_metrics = pd.read_json(PATH_M)
st.line_chart(df_metrics, x='epochs', x_label='эпохи', height=600)

## блок вывода изображения
st.header('Сравнение предсказанных и истинных признаков')
PATH_I= script_dir / 'images' / 'c_matrix_rtt.png'          # !!!ПУТЬ к изображению Confusion Matrix (поменять на свой)
st.image(PATH_I, caption=None)
