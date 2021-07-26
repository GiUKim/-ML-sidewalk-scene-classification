
from utils import *
from model import *


if __name__ == '__main__':

    model = Model_skip()
    train(exist_data=True, model=model)

    #predict_and_save()

    #evaluate()



