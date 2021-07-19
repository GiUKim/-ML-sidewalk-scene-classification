
class Config(object):
    # CNN model input size
    IMAGE_SIZE = 32
    # training 데이터셋 디렉토리 경로
    TRAIN_DIR = 'C:/Users/AI/PycharmProjects/class/datasets/train'
    # test 데이터셋 디렉토리 경로
    TEST_DIR = 'C:/Users/AI/PycharmProjects/class/dup_image_person13'
    # 클래수 개수
    NUM_CLASSES = 6
    # 클래스 종류
    CLASSES = ['cycle', 'lower', 'multi_human', 'non_human', 'one_human', 'upper']

    EPOCHS = 20

    BATCH_SIZE = 64
    # .h5 모델 경로
    MODEL_NAME = 'C:/Users/AI/PycharmProjects/class/bestmodel_20210719_111210.h5'