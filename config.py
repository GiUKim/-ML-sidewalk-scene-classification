
class Config(object):
    # CNN model input size
    IMAGE_SIZE = 32
    # training 데이터셋 디렉토리 경로
    TRAIN_DIR = 'C:/Users/AI/PycharmProjects/class/datasets/train'
    VALIDATION_DIR = 'C:/Users/AI/PycharmProjects/class/datasets/test'
    # test 데이터셋 디렉토리 경로
    #TEST_DIR = 'C:/Users/AI/PycharmProjects/class/dup_image_person13'
    #TEST_DIR = 'C:/Users/AI/PycharmProjects/class/datasets/test_prac'
    # 클래수 개수
    NUM_CLASSES = 5
    # 클래스 종류
    CLASSES = ['cycle', 'lower', 'multi_human', 'one_human', 'upper']
    # CLASSES = ['cycle', 'lower', 'multi_human', 'non_human', 'one_human', 'upper']
    #CLASSES = ['cat', 'chicken', 'dog', 'elephant', 'fly', 'horse']

    EPOCHS = 20

    BATCH_SIZE = 32
    # .h5 모델 경로
    MODEL_NAME = 'C:/Users/AI/PycharmProjects/class/model_my1_bestmodel_20210721_161456.h5'

    TEST_CYCLE = 'C:/Users/AI/PycharmProjects/class/new_classified/cycle/'
    TEST_LOWER = 'C:/Users/AI/PycharmProjects/class/new_classified/lower/'
    TEST_MULTI = 'C:/Users/AI/PycharmProjects/class/new_classified/multi_human/'
    TEST_NON = 'C:/Users/AI/PycharmProjects/class/new_classified/non_human/'
    TEST_ONE = 'C:/Users/AI/PycharmProjects/class/new_classified/one_human/'
    TEST_UPPER = 'C:/Users/AI/PycharmProjects/class/new_classified/upper/'

    # TEST_CYCLE = 'C:/Users/AI/PycharmProjects/class/new_classified/cat/'
    # TEST_LOWER = 'C:/Users/AI/PycharmProjects/class/new_classified/chicken/'
    # TEST_MULTI = 'C:/Users/AI/PycharmProjects/class/new_classified/dog/'
    # TEST_NON = 'C:/Users/AI/PycharmProjects/class/new_classified/elephant/'
    # TEST_ONE = 'C:/Users/AI/PycharmProjects/class/new_classified/fly/'
    # TEST_UPPER = 'C:/Users/AI/PycharmProjects/class/new_classified/horse/'