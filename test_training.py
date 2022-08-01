import matplotlib.pyplot as graphPlot
from segmentation_models_pytorch import Unet, encoders, utils
from torch import device, save, optim
from albumentations import Flip, ShiftScaleRotate, PadIfNeeded, RandomCrop, GaussNoise, Perspective, OneOf, \
                           CLAHE, RandomBrightnessContrast, RandomGamma, Sharpen, Blur, MotionBlur, HueSaturationValue
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from random import randrange, uniform
from json import dumps

from mask_dataset import MaskDataset
from config import trainingPath, trainingDataCoco, trainGraphicsPath, trainWidth, trainHeight
from utility import nowToString, applyCheckAugmentation, applyTrainingAugmentation, getPrepare

# выбираем, для какого устройства Torch делать модель CPU или GPU
cpuMode = False

# количество эпох обучения
trainEpochs = 25
# Максимальное количество эпох обучения, если после заданного количества эпох, модели
# продолжат улучшаться на каждом шаге обучения. В реальности это происходит 2-3 эпохи максимум
# просто любое достаточно большое число
maxTrainEpochs = 500
# порог количества эпох обучения, после которого считаем, что модель нашла (локальный) максимум и
# понижаем "скорость обучения", чтобы модель не могла покинуть этот локальный максимум
decreaseLearningRateEpochs = 20
# количество изображений, для которых одновременно обновляется количество весов
# чем больше, тем лучше, но это значение ограничено объёмом памяти видеокарты
# у меня на компьютере можно максимум 8 для resnet50
batchSize = 8

# выбранная свёрточная сеть
modelType = "resnet50"
# выбранный набор предопределённых весов
pretrainedWeights = "imagenet"
# Выбранный класс объектов. Сейчас есть разметка только для класса "building"
# но в будущем можно добавить иные классы для распознавания
classes = ["building"]


# Функция активации на последнем слое. У нас один класс, поэтому Sigmoid.
# Если делать доработку для многоклассовой модели, то функция активации
# должна стать SoftMax2D
activationFunction = "sigmoid"

# Режим поиска оптимальных параметров
searchParametersMode = False
# Количество эпох поиска оптимальных параметров
searchEpochs = 100

# в зависимости от searchParametersMode либо выбрать заранее подобранное значение,
# либо выбрать случайное значение в заданном диапазоне
def searchRange(default, minValue, maxValue, isInteger = False):
    if searchParametersMode:
        return randrange(minValue, maxValue, 1) if isInteger else round(uniform(minValue, maxValue), 2)
    else:
        return default

if __name__ == "__main__":
    # выбираем устройство torch в зависимости от настроек
    torchDevice = device("cpu" if cpuMode else "cuda")

    # подготавливаем разметку в формате COCO, хранящуюся в файле trainingDataCoco
    cocoAPI = COCO(trainingDataCoco)
    # получаем список ID категорий изображений
    categoriesIDS = cocoAPI.getCatIds()
    # получаем разметку категорий
    categories = cocoAPI.loadCats(categoriesIDS)
    # получаем список изображений, для которых размечены категории
    imagesIDs = cocoAPI.getImgIds(catIds = categoriesIDS)

    # Разделяем обучающую выборку на две части. По одной происходит обучение,
    # а другая не показывается сети, но используется для проверки качества обучения
    trainImagesIDs, checkImagesIDs = train_test_split(imagesIDs, test_size=0.1, random_state=500)

    # получить функции предварительной обработки изображения из свёрточной сети
    # и подготовить их в виде, в котором с ними сможет работать класс MaskDataset,
    # чтобы создать подготовленные изображения и подготовленные маски
    prepare = getPrepare(encoders.get_preprocessing_fn(modelType, pretrainedWeights))

    # подсчёт потерь методом "Dice" - отношение площади неправильно определённой разметки
    # к площади правильно определённой
    loss = utils.losses.DiceLoss()

    # метрика, на основании которой принимается решение о качестве обучения - IOU
    # (Intersection over Union) - аналогично Dice, только наоборот,
    # правильно определённая площадь увеличивает метрику, а неправильная - уменьшает
    # вес ложноположительной и ложноотрицательной площади - одинаков (0.5)
    metrics = [utils.metrics.IoU(threshold=0.5)]

    # поиск наибольшего значения удачности определения при подборе параметров
    searchMaxScore = 0

    # если это режим обучения, то выполняем обучение 1 раз,
    # иначе выполняем поиск оптимальных параметров заданное количество раз
    for searchParametersEpoch in range(0, 1 if not searchParametersMode else searchEpochs):
        # параметры аугментации, выбираемые по-умолчанию, если отключён режим выбора параметров,
        # либо устанавливаемые случайным образом в заданных границах
        augmentationParameters = \
        {
            # вероятности добавить изменение масштаба, поворота или сдвига
            "shiftScale": \
            {
                "probability": searchRange(0.24, 0.1, 0.8),
                "flip": searchRange(0.68, 0.1, 0.7),
                "scaleLimit": searchRange(0.64, 0.1, 0.7),
                "rotateLimit": searchRange(13, 0, 90, isInteger = True),
                "shiftLimit": searchRange(0.15, 0, 0.5)
            },

            # вероятность добавить шум по Гауссу
            "gaussNoise": searchRange(0.10, 0, 0.5),
            # вероятность внести искажение перспективой
            "perspective": searchRange(0.52, 0, 0.7),

            # вероятность применить разные преобразования контраста
            "claheBlock": \
            {
                "probability": searchRange(0.68, 0.1, 1),
                "clahe": searchRange(0.10, 0, 0.5),
                "brightnessContrast": searchRange(0.28, 0, 0.5),
                "gamma": searchRange(0.21, 0, 0.5)
            },

            # вероятность применить разные размытия и резкости
            "blurBlock": \
            {
                "probability": searchRange(0.82, 0, 1),
                "sharpen" : searchRange(0.16, 0, 0.7),
                "blurLimit": searchRange(3, 3, 5, isInteger = True),
                "blurProbability": searchRange(0.05, 0, 0.7),
                "motionBlurLimit": searchRange(4, 3, 5, isInteger = True),
                "motionBlurProbability": searchRange(0.16, 0, 0.5)
            },

            # вероятность применить смещение цвета по осям HUE
            "hueBlock": \
            {
                "probability": searchRange(0.92, 0.1, 1),
                "hueProbability": searchRange(0.20, 0.1, 1),
                "brightnessContrast": searchRange(0.20, 0.1, 1)
            }
        }

        # функция аугментации набора данных разными возможными искажениями
        # получена в результате подбора оптимальных параметров
        def trainTransform(width, height):
            return \
            [
                # вероятность добавить отражение по горизонтали или вертикали
                Flip(p = augmentationParameters["shiftScale"]["flip"]),

                # вероятности добавить изменение масштаба, поворота или сдвига
                ShiftScaleRotate(scale_limit = augmentationParameters["shiftScale"]["scaleLimit"], rotate_limit = augmentationParameters["shiftScale"]["rotateLimit"],
                                 shift_limit = augmentationParameters["shiftScale"]["shiftLimit"], p = augmentationParameters["shiftScale"]["probability"], border_mode = 0),

                # Заполнить изображения до максимального размера, если изображение меньше заданных ширины и высоты
                PadIfNeeded(min_width = width, min_height = height, always_apply = True, border_mode = 0),

                # обрезать изображение случайным образом в пределах заданной ширины и высоты
                RandomCrop(width = width, height = height, always_apply = True),

                # вероятность добавления шума
                GaussNoise(p = augmentationParameters["gaussNoise"]),
                # вероятность искажения перспективы
                Perspective(p = augmentationParameters["perspective"]),

                # случайное применение одного из преобразований в списке
                OneOf([
                        # "растягивание" гистограммы контраста для изображений, где эта гистограмма занимает не всю возможную область
                        CLAHE(p = augmentationParameters["claheBlock"]["clahe"]),
                        # случайное изменение яркости и контрастности
                        RandomBrightnessContrast(p = augmentationParameters["claheBlock"]["brightnessContrast"]),
                        # случайное изменение всех уровней гистограммы изображения
                        RandomGamma(p = augmentationParameters["claheBlock"]["gamma"]),
                      ], p = augmentationParameters["claheBlock"]["probability"]),

                OneOf([
                        # случайное добавление резкости
                        Sharpen(p = augmentationParameters["blurBlock"]["sharpen"]),
                        # случайное добавление размытия
                        Blur(blur_limit = augmentationParameters["blurBlock"]["blurLimit"], p = augmentationParameters["blurBlock"]["blurProbability"]),
                        # случайное добавление размытия в движении
                        MotionBlur(blur_limit = augmentationParameters["blurBlock"]["motionBlurLimit"], p = augmentationParameters["blurBlock"]["motionBlurProbability"]),
                      ], p = augmentationParameters["blurBlock"]["probability"]),

                OneOf([
                        # случайное изменение по шкале HUE
                        HueSaturationValue(p = augmentationParameters["hueBlock"]["hueProbability"]),
                        # случайное изменение яркости и контрастности
                        RandomBrightnessContrast(p = augmentationParameters["hueBlock"]["brightnessContrast"]),
                      ], p = augmentationParameters["hueBlock"]["probability"]),
            ]

        # Создаём модель из библиотеки segmentation_models_pytorch
        # указав выбранный тип модели для тренировки
        # и набор заданных весов
        model = Unet(
            encoder_name = modelType,
            encoder_weights = pretrainedWeights,
            # Тут пока всегда 1. Но в будущем, классов может быть несколько
            classes = len(classes),
            activation = activationFunction,
        )

        # Оптимизатор - Adam. Как-то так всегда выходит, что он оказывается лучшим
        optimizer = optim.Adam([dict(params=model.parameters(), lr=0.0001)])

        # этап обучения тренировочной выборки библиотеки segmentation_models_pytorch
        trainEpoch = utils.train.TrainEpoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=torchDevice,
            verbose=True,
        )

        # этап работы с проверочной выборкой
        # предсказание площади, для получения метрики
        checkEpoch = utils.train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics,
            device=torchDevice,
            verbose=True,
        )

        # формируем набор данных из изображений и их масок, сформированных из разметки COCO
        # все изображения проходят добавление искажений по заданным правилам для обучения,
        # а затем преобразуются в формат PyTorch
        trainDataset = MaskDataset(imagesPath=trainingPath,
                                   imagesIDs=trainImagesIDs,
                                   categoriesIDs=categoriesIDS,
                                   cocoAPI=cocoAPI,
                                   transforms=applyTrainingAugmentation(trainWidth, trainHeight, trainTransform),
                                   prepare=prepare)

        # такой же набор данных делаем для проверочной выборки
        # (так как, увы, нет возможности сравнить именно количество зданий и будут сравниваться площади заданной и
        # рассчитанной разметки зданий)
        checkDataset = MaskDataset(imagesPath=trainingPath,
                                   imagesIDs=checkImagesIDs,
                                   categoriesIDs=categoriesIDS,
                                   cocoAPI=cocoAPI,
                                   transforms=applyCheckAugmentation(trainWidth, trainHeight),
                                   prepare=prepare)

        # формируем пакетный загрузчик данных, в соответствии с задумкой авторов
        # PyTorch
        trainLoader = DataLoader(trainDataset,
                                 # количество одновременно обрабатываемых изображений
                                 batch_size=batchSize,
                                 #
                                 shuffle=True,
                                 num_workers=0,
                                 pin_memory=True,
                                 drop_last=True)

        # формируем такой же пакетный загрузчик данных для проверочной выборки
        checkLoader = DataLoader(checkDataset,
                                 batch_size=batchSize,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True,
                                 drop_last=True)

        # накапливаем список номеров эпох обучения, для оси X на графике
        trainGraphData = []

        # накапливаем метрики обучения тренировочной и проверочной выборки
        trainDiceLoss = []
        trainIOUScore = []
        checkDiceLoss = []
        checkIOUScore = []

        # значение метрики для лучшей из полученных моделей
        maxScore = 0
        # флаг, что понижение скорости обучения уже произошло, чтобы не понижать второй раз
        alreadyDecreasedLearningRate = False
        # строка, содержащая время начала обучения, для формирования имени модели
        startTimeString = nowToString()

        # если это режим обучения
        # цикл по эпохам до maxTrainEpochs,
        # но при большом maxTrainEpochs он закончится раньше
        #
        # если это режим поиска оптимальных параметров, то выполняем 2 эпохи
        for epoch in range(1, maxTrainEpochs + 1 if not searchParametersMode else 2 + 1):
            print("Epoch:", epoch)

            # выполняем этап обучения сети на тренировочной выборке
            trainLogs = trainEpoch.run(trainLoader)
            # выполняем расчёт предсказаний площади и получения метрики на проверочной выборке
            checkLogs = checkEpoch.run(checkLoader)

            # добавляем номер эпохи обучения к набору данных для оси X
            trainGraphData.append(epoch)

            # добавляем значения метрик для тренировочной и проверочной выборки
            # для формирования осей Y
            trainDiceLoss.append(trainLogs["dice_loss"])
            trainIOUScore.append(trainLogs["iou_score"])
            checkDiceLoss.append(checkLogs["dice_loss"])
            checkIOUScore.append(checkLogs["iou_score"])

            # если предыдущее лучшее значение IOU для проверочной выборки ниже текущего,
            # значит текущая модель лучше предыдущей лучшей модели
            if maxScore < checkLogs["iou_score"]:
                if not searchParametersMode:
                    # сразу сохраняем такую модель, так как обучение может идти долго,
                    # чтобы не потерять результат
                    save(model, "satellite_train_" + startTimeString + "_" + ("cpu" if cpuMode else "gpu") + ".pth")
                    # выводим информацию об улучшении метрики
                    print("Best score", format(maxScore, '.4f'), "=>", format(checkLogs["iou_score"], '.4f'), "... saved")
                # устанавливаем новое лучшее значение метрики
                maxScore = checkLogs["iou_score"]

                # если номер эпохи больше заданного порога понижения скорости обучения,
                # и этого понижения ещё не было, то понижаем скорость обучения в 10 раз.
                if epoch >= decreaseLearningRateEpochs and not alreadyDecreasedLearningRate:
                    optimizer.param_groups[0]["lr"] = 0.00001
                    alreadyDecreasedLearningRate = True

            # если текущая полученная модель не лучше предыдущей
            # и эпоха больше заданного количества эпох обучения, то прекращаем обучение
            # получается, что если модель будет улучшаться на каждом этапе, то обучение не
            # закончится до maxTrainEpochs,
            # но в реальности постоянное улучшение не идёт более 3-4 раз
            elif epoch >= trainEpochs:
                break

        if searchParametersMode:
            if maxScore > searchMaxScore:
                searchMaxScore = maxScore
                print("Best parameters:", dumps(augmentationParameters, indent = 4))

        else:
            # формируем новый график matplotlib, куда выведем графики прохождения обучения
            figure = graphPlot.figure(figsize = (14, 5))

            # добавляем слева график для значения потерь Dice
            axis1 = figure.add_subplot(1, 2, 1)
            axis1.plot(trainGraphData, trainDiceLoss, label = "train")
            axis1.plot(trainGraphData, checkDiceLoss, label = "validation")
            axis1.set_title("Dice loss")
            axis1.set_xlabel("epoch")
            axis1.set_ylabel("Dice loss")
            # подписи выводим вверху справа этого графика
            axis1.legend(loc = "upper right")

            # добавляем слева график для значения очков IOU
            axis2 = figure.add_subplot(1, 2, 2)
            axis2.plot(trainGraphData, trainIOUScore, label = "train")
            axis2.plot(trainGraphData, checkIOUScore, label = "validation")
            axis2.set_title("IOU score")
            axis2.set_xlabel("epoch")
            axis2.set_ylabel("IOU score")
            # подписи выводим вверху слева этого графика
            axis2.legend(loc = "upper left")

            # сохраняем полученный график в файл в папке trainGraphicsPath
            graphPlot.savefig(trainGraphicsPath + "/" + startTimeString + ".png")