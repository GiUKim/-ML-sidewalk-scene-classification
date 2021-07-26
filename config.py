from glob import glob

class Config(object):
    # CNN model input size
    base_dir = 'C:/Users/AI/PycharmProjects/class_2/'
    test_dir = base_dir + 'new_classfied_one_human'
    test_save_cycle = base_dir + 'new_classified/cycle'
    test_save_upper = base_dir + 'new_classified/upper'
    test_save_lower = base_dir + 'new_classified/lower'
    test_save_non = base_dir + 'new_classified/non_human'
    test_save_one = base_dir + 'new_classified/one_human'
    test_save_multi = base_dir + 'new_classified/multi_human'

    cycle_train_dir = glob(base_dir + 'datasets/train/cycle/*.jpg')
    lower_train_dir = glob(base_dir + 'datasets/train/lower/*.jpg')
    multi_train_dir = glob(base_dir + 'datasets/train/multi_human/*.jpg')
    one_train_dir = glob(base_dir + 'datasets/train/one_human/*.jpg')
    upper_train_dir = glob(base_dir + 'datasets/train/upper/*.jpg')
    non_train_dir = glob(base_dir + 'datasets/train/non_human/*.jpg')

    cycle_test_dir = glob(base_dir + 'datasets/test/cycle/*.jpg')
    lower_test_dir = glob(base_dir + 'datasets/test/lower/*.jpg')
    multi_test_dir = glob(base_dir + 'datasets/test/multi_human/*.jpg')
    one_test_dir = glob(base_dir + 'datasets/test/one_human/*.jpg')
    upper_test_dir = glob(base_dir + 'datasets/test/upper/*.jpg')
    non_test_dir = glob(base_dir + 'datasets/test/non_human/*.jpg')

    IMAGE_SIZE = 32
    threshold = 0.5
    #INPUT_SHAPE = (224, 224, 3)
    INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
    # 클래수 개수
    NUM_CLASSES = 5
    EPOCHS = 10
    BATCH_SIZE = 32
    # .h5 모델 경로
    #C:\Users\123\Desktop\ML_sidewalk - scene - classification - master\ML_sidewalk - scene - classification - master
    MODEL_NAME = base_dir + 'bestmodel_0726_1717.h5'