import os

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data

# constants
N_CLASSES = 32
TRAIN_LEN = 1357
TEST_LEN = 340
SHAPE = (1, 64, 1168)
OUT_SHAPE = (4, 32)
