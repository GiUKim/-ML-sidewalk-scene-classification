from glob import glob

class Config(object):
    # CNN model input size
    test_dir = 'C:/Users/AI/PycharmProjects/class_2/dup_image_person13'
    test_save_cycle = 'C:/Users/AI/PycharmProjects/class_2/new_classified/cycle'
    test_save_upper = 'C:/Users/AI/PycharmProjects/class_2/new_classified/upper'
    test_save_lower = 'C:/Users/AI/PycharmProjects/class_2/new_classified/lower'
    test_save_non = 'C:/Users/AI/PycharmProjects/class_2/new_classified/non_human'
    test_save_one = 'C:/Users/AI/PycharmProjects/class_2/new_classified/one_human'
    test_save_multi = 'C:/Users/AI/PycharmProjects/class_2/new_classified/multi_human'

    cycle_train_dir = glob('C:/Users/AI/PycharmProjects/class_2/datasets/train/cycle/*.jpg')
    lower_train_dir = glob('C:/Users/AI/PycharmProjects/class_2/datasets/train/lower/*.jpg')
    multi_train_dir = glob('C:/Users/AI/PycharmProjects/class_2/datasets/train/multi_human/*.jpg')
    one_train_dir = glob('C:/Users/AI/PycharmProjects/class_2/datasets/train/one_human/*.jpg')
    upper_train_dir = glob('C:/Users/AI/PycharmProjects/class_2/datasets/train/upper/*.jpg')
    non_train_dir = glob('C:/Users/AI/PycharmProjects/class_2/datasets/train/non_human/*.jpg')

    cycle_test_dir = glob('C:/Users/AI/PycharmProjects/class_2/datasets/test/cycle/*.jpg')
    lower_test_dir = glob('C:/Users/AI/PycharmProjects/class_2/datasets/test/lower/*.jpg')
    multi_test_dir = glob('C:/Users/AI/PycharmProjects/class_2/datasets/test/multi_human/*.jpg')
    one_test_dir = glob('C:/Users/AI/PycharmProjects/class_2/datasets/test/one_human/*.jpg')
    upper_test_dir = glob('C:/Users/AI/PycharmProjects/class_2/datasets/test/upper/*.jpg')
    non_test_dir = glob('C:/Users/AI/PycharmProjects/class_2/datasets/test/non_human/*.jpg')

    threshold = 0.25
    #INPUT_SHAPE = (224, 224, 3)
    INPUT_SHAPE = (32, 32, 3)
    # 클래수 개수
    NUM_CLASSES = 5
    EPOCHS = 30
    BATCH_SIZE = 32
    # .h5 모델 경로
    MODEL_NAME = 'C:/Users/AI/PycharmProjects/class_2/bestmodel_0723_1512.h5'