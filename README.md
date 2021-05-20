# Dialobot: Opensource Chatbot Framework
[![PyPI version](https://badge.fury.io/py/dialobot.svg)](https://badge.fury.io/py/dialobot)
![GitHub](https://img.shields.io/github/license/dialobot/dialobot)

![](https://user-images.githubusercontent.com/38183241/118511978-5d537180-b76d-11eb-89bd-055cb9227725.png)

## 1. What is Dialobot ?
- <u>**Opensource chatbot framework**</u> available for free.
- <u>**Neural chatbot framework**</u> using the latest models (RoBERTa, DistillUSE, mBART)
- <u>**Multilingual chatbot framework**</u> that supports English, Korean, Chinese.
- <u>**Zero-shot chatbot framework**</u> that can be used immediately without training.
- <u>**Chatbot builder**</u> that supports web application and RESTful API for services.
<br><br><br>

## 2. Installation
```console
pip install dialobot
```
<br><br>

## 3. Usage
### 3.1. Web Application
![](https://user-images.githubusercontent.com/38183241/118913444-73775280-b964-11eb-96d0-597d95a65ed1.png)
- After executing the script below, enter `localhost:FRONTEND_PORT` in your web browser to connect to the builder application.
- Since the frontend server and the backend server are running at the same time, `ctrl + c` may not shut down all server at once. At this time, make sure to shut down both servers using the `ctrl + z` + `pkill -9 Python` command.
```python
>>> from dialobot import Application
>>> Application(frontend_port=8080, backend_port=8081)
```
<br><br>

### Others
Work in process

<br><br>

## Citation
```
@misc{dialobot,
  author       = {Ko, Hyunwoong and Kim, Seonghyun and Na, Youngyun, 
                  Yang, Sooyoung, Lee, Yoonjae and Jung, Hwansuk and Oh Saechan},
  title        = {Dialobot: Opensource Chatbot Framework},
  howpublished = {\url{https://github.com/hyunwoongko/dialobot}},
  year         = {2021},
}
```

<br>

## Contributor
[Hyunwoong Ko](https://github.com/hyunwoongko), [Seonghyun Kim](https://github.com/MrBananaHuman), [Youngyun Na](https://github.com/fightnyy), [Sooyoung Yang](https://github.com/aiaaua), [Yoonjae Lee](https://github.com/gityunjae), [Hwanseok Jeong](https://github.com/jayden5744) and [Saechan Oh](https://github.com/newfull5)

<br>

## License
Dialobot project is licensed under the terms of the Apache License 2.0.

Copyright 2021 [Hyunwoong Ko](https://github.com/hyunwoongko). All Rights Reserved.

