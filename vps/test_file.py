from os import path
from sys import argv
from segmentation_models_pytorch import encoders
from PIL import Image
from torch import load, from_numpy, device

from config import trainWidth, trainHeight, userResultsPath
from mask_dataset import MaskDataset
from utility import align32, getPrepare, applyCheckAugmentation, preview, getContours


# название файла для разбора должно быть указано в командной строке, сразу после названия программы .py
if __name__ == '__main__' and len(argv) > 1:
    file = argv[1]

    # проверяем наличие файла
    if not path.isfile(file):
        print("File " + file + " not found!")
        exit(0)

    # загружаем подготовленную для слабых VPS модель
    model = load("satellite_cpu.pth")

    # выбираем выполнение работы на процессоре
    torchDevice = device("cpu")

    # получить функции предварительной обработки изображения из свёрточной сети
    # и подготовить их в виде, в котором с ними сможет работать класс MaskDataset,
    # чтобы создать подготовленные изображения и подготовленные маски
    preprocessing = getPrepare(encoders.get_preprocessing_fn("resnet50", "imagenet"))

    # создаём выборку данных из одного файла для предсказания маски
    # выполняем преобразование файла в соответствии с правилами данной программы
    # и правилами подготовки данных изображений для моделей ResNet
    checkDataset = MaskDataset(imagesPath = None,
                               imagesIDs = [file],
                               categoriesIDs = None,
                               cocoAPI = None,
                               transforms = applyCheckAugmentation(trainWidth, trainHeight),
                               prepare = preprocessing)

    # получаем изображение из подготовленного набора данных
    # нужное и единственное изображение будет под индексом 0
    image = checkDataset[0]

    # получаем тензор данных изображения в соответствии с правилами выбранного устройства torch
    # и с дополненной осью по индексу 0 с размерностью 1
    imageData = from_numpy(image).to(torchDevice).unsqueeze(0)
    # Получаем прогноз маски от модели
    # удаляем все одномерные измерения полученного тензора, так как они не несут полезной информации
    #   это всё происходит в памяти устройства torch, и данные оттуда ещё надо достать
    # cpu - переводит данные из памяти устройства torch в процессор
    # для test_file.py это не актуально, так как расчёт и так всегда производится на процессоре
    # но вдруг когда-нибудь будет делаться вариация для VPS с GPU. Тогда вызов cpu() будет нужен
    # полученные данные преобразуются в формат numpy и округляются (становятся матрицей с 0 или 1)
    predictedMask = model.predict(imageData).squeeze().cpu().numpy().round()

    # получаем список контуров зданий
    contoursPredicted = getContours(predictedMask, threshold = 0.4, minArea = 20)

    # открываем исходный файл как PIL-изображение
    imagePreviewPure = Image.open(file)
    # масштабируем его к пропорциям, кратным 32
    imagePreview = imagePreviewPure.resize((align32(imagePreviewPure.width), align32(imagePreviewPure.height)))

    # сохраняем изображение, на котором нарисованы найденные контуры в файл в папке userResultsPath
    preview(previewImage = imagePreview, contours = contoursPredicted,
            previewImageName = userResultsPath + "/" + file.split("/")[-1].replace(".jpg", ".png"))