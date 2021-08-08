# Классификация эмоций пользователей по их текстовым сообщениям на русском языке <br>

ссылка на яндекс диск с docker контейнером и другими данными: https://disk.yandex.com/d/LVoQgG1jpR3UfQ

### Инструкция:

1. Скачать zip файл с яндекс диска
2. Установить Docker и убедиться в том, что python имеет 3+ версию

#### Запуск сервера
1. Скачать файл "tensorflow-server.tar" 
2. Открыть WindowsPowerShell
3. Ввести команды <br>
  <code> docker import tensorflow-server.tar </code> <br>
  <code> tensorflow_model_server --rest_api_port=8601 --model_name=senti_model --model_base_path=/project folder/saved_models/ </code> <br>
  <code> docker run -it -v C:\path\to\your\project folder:/project folder -p 8601:8601 --entrypoint /bin/bash tensorflow/serving </code> <br>
Сервер запущен!

#### Получение предсказаний
1. Открыть cmd в той же папке где и файл "predict.py"
2. Ввести команду <br> 
  <code> python3 predict.py ваш текст для предсказания </code> <br>
  
### Информация о проекте

Результат f1: 0.80

Использовалась нн с 3-мя Bidirectional LSTM.
Были проверены разные варианты units, было решено использовать такое умножение, так как при большем возникают ошибки связанные с памятью, при меньшем результат хуже. Модель при обработке FastText не переобучается. Обучалась 24 эпохи. Когда обучалось на каждом датасете отдельно максимальный результат был (0.73, 0.92, 0.6) соответственно. Результат можно улучшить. Попробовать изменить гиперпараметры, углубить сеть, добавить механизм внимания(Attention), использовать GRU или CNN+LSTM/GRU. Не использовались предобученные сети такие как: ruBERT, SBERT и тд.
<br>
<br>
Код проекта находится в файле "TWIN-test-task-Radmir-Z.ipynb"




