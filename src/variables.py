from enum import Enum

import numpy as np


class Variable(Enum):
    SENTIMENT = 0
    SERVICE = 1
    FOOD_QUALITY = 2
    PRICE = 3
    ENVIRONMENT = 4
    RATING = 5


class Term(Enum):
    BAD = 1
    NORMAL = 2
    GOOD = 3
    LOW = 4
    HIGH = 5


sentiment = np.arange(0, 101, 1)

service = np.arange(0, 101, 1)

food_quality = np.arange(0, 101, 1)

price = np.arange(5, 61, 1)

environmnet = np.arange(0, 101, 1)

rating = np.arange(0, 101, 1)

sys_variables = [sentiment, service, food_quality, price, environmnet, rating]
