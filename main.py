
from utils import *
from model import *


if __name__ == '__main__':

    #print(model.summary())
    model6 = Model6()
    #modelg = Model_g()

    train(exist_data=True, model=model6)
    #train(exist_data=True, model=modelg)

    #evaluate()
    #predict_and_save()

