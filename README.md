# stone_detection

Установка необходимых библиотек:
```bash
pip install -r requirements
```

`inference.py` содержит класс, инициализирующий модель MaskRCNN. В классе имеется функция для получения 100 боксов и 100 масок камней соответсвенно.

Каждый бокс состоит из пяти параметров:
- первый параметр - координата ординат левой верхней точки бокса;
- второй параметр - координата абсцисс левой верхней точки бокса;
- третий параметр - координата ординат правой нижней точки бокса;
- четвертый параметр - координата абсцисс правой нижней точки бокса;
- пятый параметр - вероятность, что объект является камнем.

В `detection.ipynb` содержится пример вызова и визуализации.

`configs.py` содержит конфигурацию сети и обучения.

Необходимо скачать файл с весами модели `weights.pth` и положить рядом с выполняемым кодом.