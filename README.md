# Програмная модель FMCW радара
### Формат входных данных в файлах
* Входной файл radar_input.txt задает характеристики радара через пробел в одну строку, такие как начальные координаты (м), начальная скорость (м/с), максимальная скорость (м/с), угол направления (°), ускорение (м/с^2) и угловое ускорение (°/с)

* Входной файл object_input.txt задает характеристики в каждой строке соответствующего объекта через пробел, такие как начальные координаты (м), начальная скорость (м/с), максимальная скорость (м/с), угол направления (°), ускорение (м/с^2), угловое ускорение (°/с), ширина и длина (м)
### Установка зависимостей и запуск программы
```
pip install -r requirements.txt
python main.py
```

### Запуск демонстрации БПФ по расстоянию и допплеровского БПФ

```
python intro.py
```
