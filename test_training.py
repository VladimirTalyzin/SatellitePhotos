import matplotlib.pyplot as graphPlot
from segmentation_models_pytorch import Unet, encoders, utils
from torch import device, save, optim
from albumentations import Flip, ShiftScaleRotate, PadIfNeeded, RandomCrop, GaussNoise, Perspective, OneOf, \
                           CLAHE, RandomBrightnessContrast, RandomGamma, Sharpen, Blur, MotionBlur, HueSaturationValue
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

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
# у меня на компьютере можно максимум 8
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

    # функция аугментации набора данных разными возможными искажениями
    # получена в результате подбора оптимальных параметров
    def trainTransform(width, height):
        return \
        [
            # вероятность добавить отражение по горизонтали или вертикали
            Flip(p=0.5),

            # вероятности добавить изменение масштаба, поворота или сдвига
            ShiftScaleRotate(scale_limit = 0.35, rotate_limit = 45, shift_limit = 0.1, p = 1, border_mode = 0),

            # Заполнить изображения до максимального размера, если изображение меньше заданных ширины и высоты
            PadIfNeeded(min_width = width, min_height = height, always_apply = True, border_mode = 0),

            # обрезать изображение случайным образом в пределах заданной ширины и высоты
            RandomCrop(width = width, height = height, always_apply = True),

            # вероятность добавления шума
            GaussNoise(p = 0.1),
            # вероятность искажения перспективы
            Perspective(p = 0.4),

            # случайное применение одного из преобразований в списке
            OneOf([
                    # "растягивание" гистограммы контраста для изображений, где эта гистограмма занимает не всю возможную область
                    CLAHE(p = 1),
                    # случайное изменение яркости и контрастности
                    RandomBrightnessContrast(p = 1),
                    # случайное изменение всех уровней гистограммы изображения
                    RandomGamma(p = 1),
                  ], p=0.9),

            OneOf([
                    # случайное добавление резкости
                    Sharpen(p = 1),
                    # случайное добавление размытия
                    Blur(blur_limit = 3, p=1),
                    # случайное добавление размытия в движении
                    MotionBlur(blur_limit = 3, p = 1),
                  ], p = 0.9),

            OneOf([
                    # случайное изменение яркости и контрастности
                    RandomBrightnessContrast(p = 1),
                    # случайное изменение по шкале HUE
                    HueSaturationValue(p = 1),
                  ], p = 0.9),
        ]

    # Разделяем обучающую выборку на две части. По одной происходит обучение,
    # а другая не показывается сети, но используется для проверки качества обучения
    trainImagesIDs, checkImagesIDs = train_test_split(imagesIDs, test_size = 0.1, random_state = 500)

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

    # получить функции предварительной обработки изображения из свёрточной сети
    # и подготовить их в виде, в котором с ними сможет работать класс MaskDataset,
    # чтобы создать подготовленные изображения и подготовленные маски
    prepare = getPrepare(encoders.get_preprocessing_fn(modelType, pretrainedWeights))

    # формируем набор данных из изображений и их масок, сформированных из разметки COCO
    # все изображения проходят добавление искажений по заданным правилам для обучения,
    # а затем преобразуются в формат PyTorch
    trainDataset = MaskDataset(imagesPath = trainingPath,
                                imagesIDs = trainImagesIDs,
                                categoriesIDs = categoriesIDS,
                                cocoAPI = cocoAPI,
                                transforms = applyTrainingAugmentation(trainWidth, trainHeight, trainTransform),
                                prepare = prepare)

    # такой же набор данных делаем для проверочной выборки
    # (так как, увы, нет возможности сравнить именно количество зданий и будут сравниваться площади заданной и
    # рассчитанной разметки зданий)
    checkDataset = MaskDataset(imagesPath=trainingPath,
                                imagesIDs = checkImagesIDs,
                                categoriesIDs = categoriesIDS,
                                cocoAPI = cocoAPI,
                                transforms = applyCheckAugmentation(trainWidth, trainHeight),
                                prepare = prepare)


    # формируем пакетный загрузчик данных, в соответствии с задумкой авторов
    # PyTorch
    trainLoader = DataLoader(trainDataset,
                             # количество одновременно обрабатываемых изображений
                             batch_size = batchSize,
                             #
                             shuffle=True,
                             num_workers=0,
                             pin_memory=True,
                             drop_last=True)

    # формируем такой же пакетный загрузчик данных для проверочной выборки
    checkLoader = DataLoader(checkDataset,
                             batch_size = batchSize,
                             shuffle = False,
                             num_workers = 0,
                             pin_memory = True,
                             drop_last = True)


    # подсчёт потерь методом "Dice" - отношение площади неправильно определённой разметки
    # к площади правильно определённой
    loss = utils.losses.DiceLoss()

    # метрика, на основании которой принимается решение о качестве обучения - IOU
    # (Intersection over Union) - аналогично Dice, только наоборот,
    # правильно определённая площадь увеличивает метрику, а неправильная - уменьшает
    # вес ложноположительной и ложноотрицательной площади - одинаков (0.5)
    metrics = [ utils.metrics.IoU(threshold = 0.5) ]

    # Оптимизатор - Adam. Как-то так всегда выходит, что он оказывается лучшим
    optimizer = optim.Adam([ dict(params = model.parameters(), lr=0.0001) ])

    # этап обучения тренировочной выборки библиотеки segmentation_models_pytorch
    trainEpoch = utils.train.TrainEpoch(
        model,
        loss = loss,
        metrics = metrics,
        optimizer = optimizer,
        device = torchDevice,
        verbose = True,
    )

    # этап работы с проверочной выборкой
    # предсказание площади, для получения метрики
    checkEpoch = utils.train.ValidEpoch(
        model,
        loss = loss,
        metrics = metrics,
        device = torchDevice,
        verbose = True,
    )

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

    # цикл по эпохам до maxTrainEpochs,
    # но при большом maxTrainEpochs он закончится раньше
    for epoch in range(1, maxTrainEpochs + 1):

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
