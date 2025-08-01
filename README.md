# LSTM Team • Фаза 2 • Неделя 10
## Обработка естесственного языка • Natural Language Processing 
### Проект

__План__

- В соответствии с [инструкцией](https://github.com/Elbrus-DataScience/ds-phase-2/blob/master/08-nn/md/directory_structure.md) создайте git-репозиторий `nlp_project` и добавьте туда членов команды. 
- Разработайте [multipage](https://blog.streamlit.io/introducing-multipage-apps/)-приложение с использованием [streamlit](streamlit.io):

   - **Страница 1** • Классификация отзыва на фильм
      - [Датасет](https://drive.google.com/file/d/1c92sz81bEfOw-rutglKpmKGm6rySmYbt/view?usp=sharing) (он крайне несбалансирован, подумайте, что с этим можно сделать) содержит отзывы на фильмы, оставленные на Кинопоиске, вам необходимо построить модель классификации (поле `grade3`) введенного пользователем отзыва
      - Страница должна содержать поле ввода для пользовательского отзыва
      - Результаты предсказаний класса тремя моделями:
         - Классический ML-алгоритм, обученный на BagOfWords/TF-IDF представлении
         - RNN __или__ LSTM модель (предпочтительно использовать вариант с attention)
         - BERT-based 
       - Рядом с предсказанием модели должно выводиться время, за которое оно было получено
       - На странице должна располагаться сравнительная таблица по метрике f1-macro для всех построенных классификаторов
      
    - **Страница 2** • Оценка _степени_ токсичности пользовательского [сообщения](https://drive.google.com/file/d/1O7orH9CrNEhnbnA5KjXji8sgrn6iD5n-/view?usp=drive_link)
      - Эту задачу нужно решить с помощью модели [rubert-tiny-toxicity](https://huggingface.co/cointegrated/rubert-tiny-toxicity)
   
   - **Страница 3** • Генерация текста GPT-моделью по пользовательскому prompt
      - Данные для обучения на ваше усмотрение
      - Пользователь может регулировать длину выдаваемой последовательности
      - Число генераций
      - Температуру или `top-k/p`
     
#### Дополнительно
     
❗️Streamlit-сервис должен быть развернут на [HuggingFace Spaces](https://huggingface.co/spaces)

📊В первом задании дополните оценки моделей диаграммой attention-scores для введенных слов

> ❓[Как скачать данные с Kaggle в Google Colaboratory](https://github.com/Elbrus-DataScience/ds-phase-2/blob/master/08-nn/md/kaggle-colab.md)

> ❓[Как скачать данные с Google Drive в Google Colaboratory?](https://github.com/Elbrus-DataScience/ds-phase-2/blob/master/08-nn/md/drive-colab.md)
