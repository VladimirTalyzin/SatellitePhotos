from os import path
from sys import argv
from segmentation_models_pytorch import encoders
from PIL import Image
from torch import load, from_numpy, device

from config import trainWidth, trainHeight, userResultsPath
from mask_dataset import MaskDataset
from utility import getPrepare, applyCheckAugmentation, preview, getContours
# noinspection PyUnresolvedReferences,PyProtectedMember
from cv2 import imread, imwrite
from numpy import zeros, uint8

# Ширина изображения, поддерживаемая моделью
MODEL_WIDTH = 288
# Поддерживаемая высота изображения
MODEL_HEIGHT = 288

# Режим вычислений - на CPU или GPU
cpuMode = False
# Файл с сохранённой моделью
modelName = "satellite_gpu.pth"
# True - сохранить результат в файл, для дальнейшей обработки на сервере
# False - показать файл пользователю. Только при запуске на локальном компьютере
saveResultMode = True

# название файла для разбора должно быть указано в командной строке, сразу после названия программы .py
if __name__ == '__main__' and len(argv) > 1:
    file = argv[1]

    # проверяем наличие файла
    if not path.isfile(file):
        print("File " + file + " not found!")
        exit(0)

    # считываем указанный файл в формат библиотеки OpenCV
    image = imread(file)
    # получаем его высоту и ширину
    height, width, _ = image.shape

    # если размер изображения отличается от тех, что поддерживает модель,
    # то выполняем разбивку на части или дополнение до нужного размера
    if width != MODEL_WIDTH or height != MODEL_HEIGHT:
        tiles = []
        tilesPositions = {}
        # идём по оси Y с шагом MODEL_HEIGHT
        for row in range(0, height, MODEL_HEIGHT):
            # идём по оси X с шагом  MODEL_WIDTH
            for column in range(0, width, MODEL_WIDTH):
                # часть изображения будет сохранена во временном каталоге
                imageName = "tiles/tile_" + str(row // MODEL_HEIGHT) + "_" + str(column // MODEL_WIDTH) + ".png"
                # получаем кусочек изображения с размерами MODEL_WIDTH x MODEL_HEIGHT
                newImage = image[row:row + MODEL_HEIGHT, column:column + MODEL_WIDTH, :]
                # Измеряем, какой получился размер кусочка. Для краёв размер будет меньше
                newHeight, newWidth, _ = newImage.shape

                # сохраняем все данные о положении и размере кусочка
                tilesPositions[imageName] = {"x": column, "y": row, "width": newWidth, "height": newHeight}

                # если кусочек меньше чем MODEL_WIDTH x MODEL_HEIGHT, тогда
                # создаём черное изображение размером MODEL_WIDTH x MODEL_HEIGHT
                # и помещаем в левый верхний угол
                if newHeight < MODEL_HEIGHT or newWidth < MODEL_WIDTH:
                    blackImage = zeros((MODEL_HEIGHT, MODEL_WIDTH, 3), uint8)
                    blackImage[0:newHeight, 0:newWidth] = newImage
                    newImage = blackImage

                # сохраняем полученный кусочек
                imwrite(imageName, newImage)
                # добавляем имя файла кусочка в список картинок для обработки
                tiles.append(imageName)
    else:
        # если картинка сразу нужного размера, то никаких преобразований не производим
        # просто добавляем в список картинок для обработки
        tiles = [file]
        tilesPositions = None

    # загружаем подготовленную для слабых VPS модель
    model = load(modelName)

    # выбираем выполнение работы на процессоре
    torchDevice = device("cpu" if cpuMode else "cuda")

    # получить функции предварительной обработки изображения из свёрточной сети
    # и подготовить их в виде, в котором с ними сможет работать класс MaskDataset,
    # чтобы создать подготовленные изображения и подготовленные маски
    preprocessing = getPrepare(encoders.get_preprocessing_fn("resnet50", "imagenet"))

    # создаём выборку данных из одного файла для предсказания маски
    # выполняем преобразование файла в соответствии с правилами данной программы
    # и правилами подготовки данных изображений для моделей ResNet
    checkDataset = MaskDataset(imagesPath = None,
                               imagesIDs = tiles,
                               categoriesIDs = None,
                               cocoAPI = None,
                               transforms = applyCheckAugmentation(trainWidth, trainHeight),
                               prepare = preprocessing)

    # результирующая маска, показывающая наличие или отсутствие домов
    # если картинок для обработки было несколько, то заранее создаём файл маски, равный размеру
    # исходного изображения, заполненную черным цветом
    fullMask = None if len(tiles) == 1 else zeros((height, width), uint8)
    # получаем изображения из подготовленного набора данных
    for index, image in enumerate(checkDataset):
        # получаем тензор данных изображения в соответствии с правилами выбранного устройства torch
        # и с дополненной осью по индексу 0 с размерностью 1
        imageData = from_numpy(image).to(torchDevice).unsqueeze(0)
        # Получаем прогноз маски от модели
        # удаляем все одномерные измерения полученного тензора, так как они не несут полезной информации
        #   это всё происходит в памяти устройства torch, и данные оттуда ещё надо достать
        # cpu - переводит данные из памяти устройства torch в процессор
        # для test_file.py это не актуально, так как расчёт и так всегда производится на процессоре,
        # но вдруг когда-нибудь будет делаться вариация для VPS с GPU. Тогда вызов cpu() будет нужен
        # полученные данные преобразуются в формат numpy, умножаем на подобранный коэффициент
        # и округляем
        predictedMask = (model.predict(imageData).squeeze().cpu().numpy() * 10).round()

        # если был всего один кусочек для обработки, то сохраняем полученную маску как результат
        if len(tiles) == 1:
            fullMask = predictedMask

        # если же кусочков было несколько, тогда собираем большую маску из нескольких маленьких
        else:
            # получаем размеры и положение кусочка файла
            positions = tilesPositions[tiles[index]]
            savedX = positions["x"]
            savedY = positions["y"]
            savedWidth  = positions["width"]
            savedHeight = positions["height"]

            # получаем размеры полученной от модели маски (должно быть MODEL_WIDTH x MODEL_HEIGHT)
            maskHeight, maskWidth = predictedMask.shape

            # копируем полученную маску в нужную позицию на большой маске
            fullMask[savedY : savedY + savedHeight,
                     savedX : savedX + savedWidth] = predictedMask \
                        if maskWidth == savedWidth and maskHeight == savedHeight else predictedMask[0 : savedHeight, 0 : savedWidth]
                        # а если маска выходит за пределы изображения, тогда обрезаем справа и снизу

    # получаем список контуров зданий
    contoursPredicted = getContours(fullMask, threshold = 0.9, minArea = 0)

    # открываем исходный файл как PIL-изображение
    imagePreview = Image.open(file)

    # сохраняем или показываем изображение, на котором нарисованы найденные контуры в файл в папке userResultsPath
    preview(previewImage = imagePreview, contours = contoursPredicted,
            previewImageName = userResultsPath + "/" + file.split("/")[-1].replace(".jpg", ".png"), saveOrShow = saveResultMode)