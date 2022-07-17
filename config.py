# папка с изображениями для обучения
trainingPath = "train"
# coco-разметка данных изображений для обучения
trainingDataCoco = "coco-training.json"
# список файлов в папке для обучения и количество зданий на каждом из них
trainingData = "train.csv"

# папка с изображениями для определения зданий
checkPath = "check"
# список файлов в папке
checkData = "sample_solution.csv"

# файл, куда записывается результат, который можно отправлять на чемпионат
resultData = "result.csv"

# папка, куда записываются преобразованные пользовательские файлы, отправленные на VPS
userResultsPath = "userResults"

# папка, куда записываются графики метрик обучения
trainGraphicsPath = "graphs"

# 300x300 - размер картинок в обучающей выборке
trainWidth  = 300
trainHeight = 300