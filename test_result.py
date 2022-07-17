from csv import writer, DictReader
import segmentation_models_pytorch as smp
from PIL import Image
from numpy import arange
from torch import load, from_numpy, device
from tqdm import tqdm

from mask_dataset import MaskDataset
from config import checkPath, checkData, trainingPath, trainingData, trainWidth, trainHeight, resultData
from utility import align32, getPrepare, applyCheckAugmentation, preview, getContours

# режим работы
# True - оптимизация параметров
# False - получение контуров на фотографиях и формирование csv-файла результата
scoreTrainMode = False

# файл с обученной моделью
modelName = "satellite_gpu.pth"
# выбранная свёрточная сеть
modelType = "resnet50"
# выбранный набор предопределённых весов
pretrainedWeights = "imagenet"

if __name__ == "__main__":
    # считываем заранее подготовленную модель из файла
    model = load(modelName)

    # если в названии модели есть слово cpu, то выбирается устройство torch CPU,
    # иначе работа ведётся на GPU
    torchDevice = device("cpu" if "cpu" in modelName else "cuda")

    # получить функции предварительной обработки изображения из свёрточной сети
    # и подготовить их в виде, в котором с ними сможет работать класс MaskDataset,
    # чтобы создать подготовленные изображения и подготовленные маски
    prepare = getPrepare(smp.encoders.get_preprocessing_fn(modelType, pretrainedWeights))

    # выполнить расчёт количества зданий с заданными параметрами
    # в режиме savePreview = True, будут также сохранены фотографии с нанесёнными контурами домов
    # threshold - порог очерчивания контура
    # minArea - минимальная площадь, контуры ниже которой считаются ошибкой и отбрасываются
    def prediction(ids, dataset, threshold, minArea, savePreview):
        result = {}
        # выполняем для всего набора фотографий, выводя красивый ProgressBar
        for imageIndex in tqdm(range(len(ids))):
            imageName = ids[imageIndex]
            image = dataset[imageIndex]

            # получаем тензор данных изображения в соответствии с правилами выбранного устройства torch
            # и с дополненной осью по индексу 0 с размерностью 1
            imageData = from_numpy(image).to(torchDevice).unsqueeze(0)
            # Получаем прогноз маски от модели
            # удаляем все одномерные измерения полученного тензора, так как они не несут полезной информации
            #   это всё происходит в памяти устройства torch, и данные оттуда ещё надо достать
            # cpu - переводит данные из памяти устройства torch в процессор, если это было GPU-устройство
            # если устройство и было cpu, то просто возвращаются те же данные
            # полученные данные преобразуются в формат numpy и округляются (становятся матрицей с 0 или 1)
            predictedMask = model.predict(imageData).squeeze().cpu().numpy().round()

            # получаем список контуров
            contoursPredicted = getContours(predictedMask, threshold = threshold, minArea = minArea)

            # количество зданий равно количеству контуров
            result[imageName] = len(contoursPredicted)

            # если режим работы с сохранением фото с контурами
            if savePreview:
                # открываем изображение в формате библиотеки PIL
                imagePreviewPure = Image.open(checkPath + "/" + imageName)
                # масштабируем до ближайших меньших размеров, кратных 32
                imagePreview = imagePreviewPure.resize((align32(imagePreviewPure.width), align32(imagePreviewPure.height)))
                # сохраняем изображение и контуры в файл
                preview(previewImage = imagePreview, contours = contoursPredicted, previewImageName=imageName)

        # возвращаем накопленные значения зданий для картинок
        return result


    if scoreTrainMode:
        # считываем тренировочные данные, для определения качества определения количества домов
        # и оптимизации параметров
        trainData = {}
        with open(trainingData, "r") as csvFile:
            reader = DictReader(csvFile)
            for row in reader:
                imageName = row["img_num"]
                trainData[imageName] = int(row["number_of_houses"])

        # список изображений равен списку ключей в trainData
        trainIDs = list(trainData.keys())
        # формируем набор данных для проверки работы предсказания количества зданий
        # указывается просто список изображений, для которых никаких масок не формируется,
        # поэтому categoriesIDs и cocoAPI равный None
        # изображения проходят подготовку albumentations для тестируемых изображений и
        # затем преобразуются в формат PyTorch
        trainDataset = MaskDataset(imagesPath = trainingPath,
                                   imagesIDs = trainIDs,
                                   categoriesIDs = None,
                                   cocoAPI = None,
                                   transforms = applyCheckAugmentation(trainWidth, trainHeight),
                                   prepare = prepare)

        # аккумулируем лучший вариант учёта
        bestLoss = None

        # вариации порог очерчивания контура
        for threshold in arange(0.0, 1.0, 0.1):
            # вариации порога отбрасывания малых контуров
            for minArea in range(5, 50, 5):
                # получаем предсказанное количество зданий для тренировочной выборки
                # которое можно будет сравнить с известными значениями, для
                # подсчёта количества потерь
                result = prediction(ids = trainIDs, dataset = trainDataset, threshold = threshold, minArea = minArea, savePreview = False)

                # суммируем разницу между значением в обучающей выборке и предсказанными значениями
                # чем меньше будет сумма различий, тем лучшие параметры найдены
                loss = 0
                for imageName, validNumbers in trainData.items():
                    loss += abs(validNumbers - result[imageName])

                if bestLoss is None or bestLoss > loss:
                    # выводим информацию о том, как понизился
                    print("Best score", "None" if bestLoss is None else str(bestLoss), "=>", loss)
                    # выводим лучший набор параметров
                    print("Best threshold:", threshold)
                    print("Best minArea:", minArea)
                    # сохраняем текущее лучшее значение потери
                    bestLoss = loss


    else:
        # считываем заданный список зданий для определения количества и построения контуров
        imagesIDs = []
        with open(checkData, "r") as csvFile:
            reader = DictReader(csvFile)
            for row in reader:
                imagesIDs.append(row["img_num"])

        # формируем набор данных из изображений, для которых нужно рассчитать количество зданий и отобразить контуры
        # указывается просто список изображений, для которых никаких масок не формируется,
        # поэтому categoriesIDs и cocoAPI равный None
        # изображения проходят подготовку albumentations для тестируемых изображений и
        # затем преобразуются в формат PyTorch
        checkDataset = MaskDataset(imagesPath = checkPath,
                                   imagesIDs = imagesIDs,
                                   categoriesIDs = None,
                                   cocoAPI = None,
                                   transforms = applyCheckAugmentation(trainWidth, trainHeight),
                                   prepare = prepare)

        # получаем предсказанное количество зданий и сохраняем контуры на фотографиях
        result = prediction(ids = imagesIDs, dataset = checkDataset, threshold = 0.3, minArea = 10, savePreview = True)

        # записываем полученное количество зданий в файл результата
        # очерёдность записи делаем такой же, как у файла checkData
        with open(resultData, "w", newline = "") as csvFile:
            csvWriter = writer(csvFile, delimiter = ",")
            csvWriter.writerow(["img_num", "number_of_houses"])
            for imageName in imagesIDs:
                csvWriter.writerow([imageName, result[imageName]])
