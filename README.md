# SatellitePhotos

test_training.py - запустить обучение и сохранить лучшую модель

test_result.py - запустить распознавание по сохранённой модели

test_file.py - запустить распознавание файла, указанного в командной строке. Пример:

```
cd /var/www/morfo/data/www/0v.ru/satellite/
python test_file.py 1.jpg
```

А также, если сразу запускать распознавание, сначала необходимо скачать файл модели:
* Для GPU: https://0v.ru/satellite/satellite_gpu.pth
* Для CPU: https://0v.ru/satellite/satellite_cpu.pth


Зависимости для тренировки моделей:

* pip install torch-gpu
* pip install torchvision
* pip install albumentations
* pip install segmentation-models-pytorch
* pip install pycocotools
* pip install opencv-python
* pip install numpy
* pip install scikit-learn
* pip install Pillow
* pip install tqdm

# Установка на VPS

Пример работы сайта, размещённого на VPS: https://0v.ru/satellite/

Необходимо установить на VPS содержимое папки "VPS". 
Туда же скачать файл модели: https://0v.ru/satellite/satellite_cpu.pth
На все папки необходимо установить права 777.

Работает на любых платформах. Desktop, Android, iOS

![Запуск сайта](https://0v.ru/satellite/screen-1.png)
![Здания найдены](https://0v.ru/satellite/screen-2.png)
![Здания найдены](https://0v.ru/satellite/screen-3.png)

Зависимости для запуска на VPS:

* pip install torchvision
* pip install albumentations
* pip install segmentation-models-pytorch
* pip install opencv-python
* pip install numpy
* pip install Pillow
* apt-get install libgl1  

