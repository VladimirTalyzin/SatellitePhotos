from datetime import datetime
from albumentations import Lambda, Compose, Resize
from matplotlib.pyplot import figure, imshow, plot, savefig, close, box, axis, show
from numpy import expand_dims, float32
from skimage import measure
import cv2

# найти ближайшее меньшее число, делящееся на 32
# дробность к 32 нужна для того, чтобы сохранить как можно больше данных
# при свёртке изображения
def align32(value):
    return int(value / 32) * 32


# получить текущее значение времени, пригодное для использования в названиях файлов
def nowToString():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


# объединить заданные команды добавления изображений в обучающую выборку
# по правилам библиотеки albumentations
def applyTrainingAugmentation(width, height, transform):
    return Compose(transform(align32(width), align32(height)))


# получить команду масштабирования тестируемых изображений
# до размера, кратного 32,
# по правилам библиотеки albumentations
def applyCheckAugmentation(width, height):
    return Compose([Resize(align32(width), align32(height))])


# получить команду подготовки изображения с помощью библиотеки albumentations
# в формат, пригодный для PyTorch
def getPrepare(prepareFunction):
    # передача функции преобразования тензора albumentations
    # в тензор PyTorch. Меняются оси х, y и rgb
    # noinspection PyUnusedLocal
    def toTensor(value, **kwargs):
        return value.transpose(2, 0, 1).astype("float32")

    return Compose([
        Lambda(image = prepareFunction),
        Lambda(image = toTensor, mask = toTensor)
    ])


# вывод изображения с нарисованными поверх контурами
# showOrSave = True - сохранение в файл
# showOrSave = False - показать файл пользователю
def preview(previewImage, contours, previewImageName, saveOrShow = True):
    dpi = 100
    figureSize = (previewImage.width / dpi, previewImage.height / dpi)
    # создаём фигуру matplotlib
    previewFigure = figure(figsize = figureSize, dpi = dpi)
    # убираем рамки вокруг и оси, они тут не нужны
    box(False)
    axis('off')

    # вывести изображение на графике
    imshow(previewImage)

    # вывести все контуры разными цветами
    for number, contour in enumerate(contours):
        plot(contour[:, 1], contour[:, 0], linewidth=2)

    # сохранить полученное изображение в PNG-файл
    if saveOrShow:
        savefig(previewImageName if "/" in previewImageName else "results/preview_" + previewImageName.replace(".jpg", ".png"),
            bbox_inches = "tight", pad_inches = 0)
    else:
        show()

    # Удалить объект фигуры из памяти.
    # При формировании 900 картинок проблема утечки памяти становится актуальной
    close(previewFigure)


# найти все контуры на двумерной маске
# threshold - порог "склеивания" рядом стоящих контуров
# minArea - контуры, площадь которых меньше указанной,
# считаются ошибкой и отбрасываются
def getContours(mask, threshold, minArea):
    contours = measure.find_contours(mask, threshold)
    contoursResult = []
    for contour in contours:
        # добавляем ещё одну ось к данным контуров, чтобы они соответствовали формату OpenCV
        contourWithAxis = expand_dims(contour.astype(float32), 1)
        # получаем класс UMat OpenCV для контура
        # noinspection PyUnresolvedReferences
        umatContour = cv2.UMat(contourWithAxis)
        # получаем площадь контура
        # noinspection PyUnresolvedReferences
        area = cv2.contourArea(umatContour)
        if area > minArea:
            contoursResult.append(contour)

    return contoursResult