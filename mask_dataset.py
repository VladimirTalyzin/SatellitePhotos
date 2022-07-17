from torch.utils.data import Dataset
from PIL import Image
from numpy import array, zeros, maximum

# класс, хранящий набор изображений в формате, необходимым PyTorch для работы
# и формирующий маску из разметки COCO, если такая разметка была задана
class MaskDataset(Dataset):
    # imagesPath - папка, где лежат все изображения из создаваемого набора. Может быть None,
    # тогда имена файлов должны содержать полный путь к ним
    #
    # Возможны два варианта инициализации
    #
    # imagesIDs - набор данных об изображениях в формате COCO
    # categoriesIDs - набор данных о разметке категорий изображений в формате COCO
    # cocoAPI - объект COCO
    #
    # либо (в этом случае маска изображений не создаётся)
    #
    # imagesIDs - список имён файлов в папке imagesPath
    # categoriesIDs - None
    # cocoAPI - None
    #
    # transforms - правила преобразования изображений из библиотеки albumentations
    # prepare - функция конвертации из albumentations в формат Torch
    def __init__(self, imagesPath, imagesIDs, categoriesIDs, cocoAPI, transforms, prepare):
        self.imagesPath = imagesPath
        self.imagesIDs = imagesIDs
        self.categoriesIDs = categoriesIDs
        self.cocoAPI = cocoAPI
        self.transforms = transforms
        self.prepare = prepare

    # функция перечисления для итератора
    def __getitem__(self, index):
        imageID = self.imagesIDs[index]

        # если при инициализации выбран вариант простого набора изображений
        if self.cocoAPI is None:
            # тогда имя файла просто содержится в списке данных
            fileName = imageID
            # полный путь к файлу изображения
            fileWithPath = (self.imagesPath + "/" if self.imagesPath is not None else "") + fileName
            # прочитать изображение библиотекой PIL и преобразовать в формат, когда каждый пиксель
            # представлен в виде трёх чисел R, G, B
            image = Image.open(fileWithPath).convert("RGB")
            # применить правила аугментации albumentations
            # чтобы не делать отдельный функционал, в качестве маски передаётся тоже самое изображение
            # но результат маски игнорируется
            # noinspection PyTypeChecker
            augmented = self.transforms(image = array(image), mask = array(image))
            image = augmented["image"]
            # подготавливаем изображение к формату PyTorch
            augmented = self.prepare(image=image, mask=array(image))

            return augmented["image"]

        # если при инициализации выбрано создание маски по данным разметки COCO
        else:
            # получить информацию об изображении COCO
            imageCOCOInfo = self.cocoAPI.loadImgs(imageID)[0]
            # получить имя файла
            fileName = imageCOCOInfo["file_name"]
            # полный путь к файлу изображения
            fileWithPath = (self.imagesPath + "/" if self.imagesPath is not None else "") + fileName
            # прочитать изображение библиотекой PIL и преобразовать в формат, когда каждый пиксель
            # представлен в виде трёх чисел R, G, B
            image = Image.open(fileWithPath).convert("RGB")
            # создать маску по данным разметки COCO
            mask = self.__getMask(self.cocoAPI, imageCOCOInfo, imageID, self.categoriesIDs)

            # применить правила аугментации albumentations к изображению и маске
            # noinspection PyTypeChecker
            augmented = self.transforms(image = array(image), mask = mask)
            image = augmented["image"]
            mask = augmented["mask"]

            # подготовить изображение и маску к формату PyTorch
            augmented = self.prepare(image = image, mask = mask)

            return augmented["image"], augmented["mask"]


    # длина набора изображений для итератора
    def __len__(self):
        return len(self.imagesIDs)


    # создание маски изображения для категорий COCO
    def __getMask(self, coco, imageCOCOInfo, imageID, categoriesIDs):
        # получить ID разметок для указанного изображения
        annotationsIDs = self.cocoAPI.getAnnIds(imgIds = imageID,
                                                catIds = categoriesIDs,
                                                iscrowd = None)

        # получить саму разметку
        annotations = coco.loadAnns(annotationsIDs)


        # создаём маски заданной ширины и высоты для каждой категории (у нас пока 1) и заполняем нулями
        imageData = (imageCOCOInfo["width"], imageCOCOInfo["height"], len(categoriesIDs))
        masks = zeros(imageData)

        # для всех возможных категорий формируем маски
        # у нас категория одна - здания, но оставлена возможность
        # для введения многоклассовой работы
        for index, categoryID in enumerate(categoriesIDs):
            # создаём маску заданной ширины и высоты и заполняем нулями
            mask = zeros(imageData[:2])
            # проходимся по всей разметке изображения
            for annotation in annotations:
                # и если совпадает категория
                if categoryID == annotation["category_id"]:
                    # то заполняем обозначенные здания значением, предлагаемым COCO (то есть единицами)
                    mask = maximum(mask, self.cocoAPI.annToMask(annotation))

            # записываем маску в слот её категории
            masks[:, :, index] = mask

        return masks