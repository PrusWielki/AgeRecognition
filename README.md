# AgeRecognition

### Pobranie repozytorium

Na samym początku pobieramy repozytorium:
```
git clone https://github.com/HaZeL-spc/AgeRecognition.git
```
**Wszystkie** następne komendy zapisujemy w katalogu repozytorium pobranego na naszym komputerze

### Instalacja pythona 3.11.0

Tutaj sposób instalacji będzie zależał od systemu operacyjnego, generalnie pod tym linkiem można się dowiedzieć jak zainstalować pythona na swojej maszynie ⇒ https://www.python.org/  
**Uwaga:** Python powinien być pobrany w odpowiedniej wersji (3.11.0) oraz razem z pythonem należy zainstalować pip, czyli menadżer pakietów pythona.

Aby sprawdzić czy python i pip zostały poprawnie zainstalowane należy wpisać w konsolę komendy:
```python --version``` oraz ```pip --version```.
Output powinien wyglądać tak:
```
> python --version
Python 3.11.0
> pip --version
pip x.x.x from ... (python 3.11)
```

### Instalacja virtualenva

Należy wpisać komendę:
```
pip install virtualenv
```

### Utworzyć środowisko wirtualne

Jeśli Python --version nie daje nam 3.11.0 wpisuję ścieżkę do python.exe zamiast python np.:
```
> C:\Users\MyUser\AppData\Local\Programs\Python\Python311\python.exe -m venv my_env
```

Natomiast w przeciwnym wypadku należy wykonać komendę:
```
> python -m venv my_env
```


Aby sprawdzić czy środowisko wirtualne działa, należy aktywować je poleceniem:
Linux:
```
> source my_env/bin/activate
```
lub
Windows:
```
> .\my_env\Scripts\activate
```

W wyniku aktywacji przed znakiem zachęty w konsoli powinna pojawić się nazwa venva:
```
(my_env) >
```

### Instalacja potrzebnych bibliotek

W katalogu, w którym się znajdujecie jest plik `requirements.txt`. Zawiera on wszystkie potrzebne do przeprowadzenia warsztatów biblioteki. Aby je zainstalować nalezy wywołać komendę:
```
(my_env) > pip install -r requirements.txt
```

## 1. Instrukcja dla użytkownika

Żeby włączyć program musimy użyć komendy

```
(my_env) > python main.py
```

## 2. Instrukcja dla developera

### Instalacja kernela dla utworzonego środowiska

Przy aktywowanym środowisku należy wywołać komendę:
```
(my_env) > ipython kernel install --user --name=my_env
```
Po tej operacji zostanie utworzony kernel jupytera dla środowiska wirtualnego.

## Sprawdzenie czy konfiguracja środowiska przebiegła pomyślnie
Należy wywołać komendę w głównym katalogu repozytorium:
```
(my_env) > jupyter notebook
```

Następnie wybieramy nasz jupyter plik i w nim pracujemy