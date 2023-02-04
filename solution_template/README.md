# Подготовка к работе

* Установите C++ компилятор
* Установите Python библиотеки, например с помощью pip

```shell
pip3 install -r requirements.txt
```

* Выполните сборку вашей C++ библиотеки

```shell
python3 setup.py build_ext -i
```

* Проверьте, что библиотека проходит тесты

```shell
pytest run_tests.py
```

Описание задания находится в ноутбуке `motion_estimation.ipynb`.
Начните с его прочтения и исследования файла `me_estimator.cpp`.

# Сдача задания

В cv-gml решение будет запускаться с помощью скриптов из папки `tests`.
Перед отправкой проверьте, что ваше решение корректно тестируется с их помощью

```shell
python3 -m tests.compose
python3 -m tests.run
```

# Вопросы

Если у вас не заработало что-то или есть вопросы о C++ шаблоне, смело пишите в чатик курса