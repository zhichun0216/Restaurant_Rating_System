from collections import namedtuple
from functools import reduce

import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt

from src import MF, sys_variables, Variable, Term

T = namedtuple("Variable", ['name', 'term'])


class Rule:

    def __init__(self, IF, THEN: T):
        self.MFs = IF
        self.outputMF = THEN

        self.mfs = [MF[m.name][m.term] for m in IF]
        self.output_mf = MF[THEN.name][THEN.term]

    def predict(self, inputs: map) -> np.ndarray:

        inputs = [inputs[m.name] for m in self.MFs]

        self.output = []

        for i in range(len(inputs)):
            v = self.MFs[i]
            xmf = self.mfs[i]
            xx = inputs[i]

            x = sys_variables[v.name.value]

            self.output.append(fuzz.interp_membership(x, xmf, xx))

        self.clip = np.fmin(self.output_mf, reduce(min, self.output))
        return self.clip

    def __str__(self) -> str:
        antecedence = ' and '.join([f"[{m.name.name}] is [{m.term.name}]" for m in self.MFs])
        consequence = f'[{self.outputMF.name.name}] is [{self.outputMF.term.name}].'
        return "If {:s}, then {:s}".format(antecedence, consequence)

    def visualize(self):
        fig, ax = plt.subplots(1, (len(self.MFs) + 1), figsize=(15, 5))

        last = len(ax) - 1
        curve = self.output.copy()
        curve.append(self.clip)
        mfs = [m.name for m in self.MFs]
        mfs.append(Variable.RATING)

        for i, var in enumerate(mfs):

            for item, mf in MF[var].items():
                ax[i].plot(mf)

            ax[i].set_title(var.name)
            ax[i].legend(MF[var].keys(), loc='upper right')

            if i != last:
                ax[i].plot([curve[i] for _ in sys_variables[var.value]])
            else:
                ax[i].plot(curve[i])
                ax[i].fill_between(sys_variables[var.value], curve[i])

        plt.show()


rules = [
    # Sentiment vs others
    Rule(IF=[T(Variable.SENTIMENT, Term.BAD), T(Variable.SERVICE, Term.BAD)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.SENTIMENT, Term.BAD), T(Variable.SERVICE, Term.NORMAL)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.SENTIMENT, Term.BAD), T(Variable.SERVICE, Term.GOOD)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.SENTIMENT, Term.BAD), T(Variable.FOOD_QUALITY, Term.BAD)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.SENTIMENT, Term.BAD), T(Variable.FOOD_QUALITY, Term.NORMAL)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.SENTIMENT, Term.BAD), T(Variable.FOOD_QUALITY, Term.GOOD)], THEN=T(Variable.RATING, Term.GOOD)),
    Rule(IF=[T(Variable.SENTIMENT, Term.BAD), T(Variable.PRICE, Term.LOW)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.SENTIMENT, Term.BAD), T(Variable.PRICE, Term.NORMAL)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.SENTIMENT, Term.BAD), T(Variable.PRICE, Term.HIGH)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.SENTIMENT, Term.BAD), T(Variable.ENVIRONMENT, Term.BAD)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.SENTIMENT, Term.BAD), T(Variable.ENVIRONMENT, Term.NORMAL)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.SENTIMENT, Term.BAD), T(Variable.ENVIRONMENT, Term.GOOD)], THEN=T(Variable.RATING, Term.NORMAL)),

    Rule(IF=[T(Variable.SENTIMENT, Term.NORMAL), T(Variable.SERVICE, Term.BAD)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.SENTIMENT, Term.NORMAL), T(Variable.SERVICE, Term.NORMAL)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.SENTIMENT, Term.NORMAL), T(Variable.SERVICE, Term.GOOD)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.SENTIMENT, Term.NORMAL), T(Variable.FOOD_QUALITY, Term.BAD)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.SENTIMENT, Term.NORMAL), T(Variable.FOOD_QUALITY, Term.NORMAL)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.SENTIMENT, Term.NORMAL), T(Variable.FOOD_QUALITY, Term.GOOD)], THEN=T(Variable.RATING, Term.GOOD)),
    Rule(IF=[T(Variable.SENTIMENT, Term.NORMAL), T(Variable.PRICE, Term.LOW)], THEN=T(Variable.RATING, Term.GOOD)),
    Rule(IF=[T(Variable.SENTIMENT, Term.NORMAL), T(Variable.PRICE, Term.NORMAL)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.SENTIMENT, Term.NORMAL), T(Variable.PRICE, Term.HIGH)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.SENTIMENT, Term.NORMAL), T(Variable.ENVIRONMENT, Term.BAD)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.SENTIMENT, Term.NORMAL), T(Variable.ENVIRONMENT, Term.NORMAL)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.SENTIMENT, Term.NORMAL), T(Variable.ENVIRONMENT, Term.GOOD)], THEN=T(Variable.RATING, Term.GOOD)),

    Rule(IF=[T(Variable.SENTIMENT, Term.GOOD), T(Variable.SERVICE, Term.BAD)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.SENTIMENT, Term.GOOD), T(Variable.SERVICE, Term.NORMAL)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.SENTIMENT, Term.GOOD), T(Variable.SERVICE, Term.GOOD)], THEN=T(Variable.RATING, Term.GOOD)),
    Rule(IF=[T(Variable.SENTIMENT, Term.GOOD), T(Variable.FOOD_QUALITY, Term.BAD)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.SENTIMENT, Term.GOOD), T(Variable.FOOD_QUALITY, Term.NORMAL)], THEN=T(Variable.RATING, Term.GOOD)),
    Rule(IF=[T(Variable.SENTIMENT, Term.GOOD), T(Variable.FOOD_QUALITY, Term.GOOD)], THEN=T(Variable.RATING, Term.GOOD)),
    Rule(IF=[T(Variable.SENTIMENT, Term.GOOD), T(Variable.PRICE, Term.LOW)], THEN=T(Variable.RATING, Term.GOOD)),
    Rule(IF=[T(Variable.SENTIMENT, Term.GOOD), T(Variable.PRICE, Term.NORMAL)], THEN=T(Variable.RATING, Term.GOOD)),
    Rule(IF=[T(Variable.SENTIMENT, Term.GOOD), T(Variable.PRICE, Term.HIGH)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.SENTIMENT, Term.GOOD), T(Variable.ENVIRONMENT, Term.BAD)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.SENTIMENT, Term.GOOD), T(Variable.ENVIRONMENT, Term.NORMAL)], THEN=T(Variable.RATING, Term.GOOD)),
    Rule(IF=[T(Variable.SENTIMENT, Term.GOOD), T(Variable.ENVIRONMENT, Term.GOOD)], THEN=T(Variable.RATING, Term.GOOD)),

    # # Service vs others
    Rule(IF=[T(Variable.SERVICE, Term.BAD), T(Variable.FOOD_QUALITY, Term.BAD)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.SERVICE, Term.BAD), T(Variable.FOOD_QUALITY, Term.NORMAL)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.SERVICE, Term.BAD), T(Variable.FOOD_QUALITY, Term.GOOD)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.SERVICE, Term.BAD), T(Variable.PRICE, Term.LOW)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.SERVICE, Term.BAD), T(Variable.PRICE, Term.NORMAL)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.SERVICE, Term.BAD), T(Variable.PRICE, Term.HIGH)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.SERVICE, Term.BAD), T(Variable.ENVIRONMENT, Term.BAD)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.SERVICE, Term.BAD), T(Variable.ENVIRONMENT, Term.NORMAL)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.SERVICE, Term.BAD), T(Variable.ENVIRONMENT, Term.GOOD)], THEN=T(Variable.RATING, Term.NORMAL)),

    Rule(IF=[T(Variable.SERVICE, Term.NORMAL), T(Variable.FOOD_QUALITY, Term.BAD)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.SERVICE, Term.NORMAL), T(Variable.FOOD_QUALITY, Term.NORMAL)], THEN=T(Variable.RATING, Term.NORMAL)), 
    Rule(IF=[T(Variable.SERVICE, Term.NORMAL), T(Variable.FOOD_QUALITY, Term.GOOD)], THEN=T(Variable.RATING, Term.GOOD)),
    Rule(IF=[T(Variable.SERVICE, Term.NORMAL), T(Variable.PRICE, Term.LOW)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.SERVICE, Term.NORMAL), T(Variable.PRICE, Term.NORMAL)], THEN=T(Variable.RATING, Term.NORMAL)), 
    Rule(IF=[T(Variable.SERVICE, Term.NORMAL), T(Variable.PRICE, Term.HIGH)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.SERVICE, Term.NORMAL), T(Variable.ENVIRONMENT, Term.BAD)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.SERVICE, Term.NORMAL), T(Variable.ENVIRONMENT, Term.NORMAL)], THEN=T(Variable.RATING, Term.NORMAL)), 
    Rule(IF=[T(Variable.SERVICE, Term.NORMAL), T(Variable.ENVIRONMENT, Term.GOOD)], THEN=T(Variable.RATING, Term.NORMAL)),

    Rule(IF=[T(Variable.SERVICE, Term.GOOD), T(Variable.FOOD_QUALITY, Term.BAD)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.SERVICE, Term.GOOD), T(Variable.FOOD_QUALITY, Term.NORMAL)], THEN=T(Variable.RATING, Term.GOOD)),
    Rule(IF=[T(Variable.SERVICE, Term.GOOD), T(Variable.FOOD_QUALITY, Term.GOOD)], THEN=T(Variable.RATING, Term.GOOD)),
    Rule(IF=[T(Variable.SERVICE, Term.GOOD), T(Variable.PRICE, Term.LOW)], THEN=T(Variable.RATING, Term.GOOD)),
    Rule(IF=[T(Variable.SERVICE, Term.GOOD), T(Variable.PRICE, Term.NORMAL)], THEN=T(Variable.RATING, Term.GOOD)),
    Rule(IF=[T(Variable.SERVICE, Term.GOOD), T(Variable.PRICE, Term.HIGH)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.SERVICE, Term.GOOD), T(Variable.ENVIRONMENT, Term.BAD)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.SERVICE, Term.GOOD), T(Variable.ENVIRONMENT, Term.NORMAL)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.SERVICE, Term.GOOD), T(Variable.ENVIRONMENT, Term.GOOD)], THEN=T(Variable.RATING, Term.GOOD)),

    # # Food quality vs others
    Rule(IF=[T(Variable.FOOD_QUALITY, Term.BAD), T(Variable.PRICE, Term.LOW)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.FOOD_QUALITY, Term.BAD), T(Variable.PRICE, Term.NORMAL)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.FOOD_QUALITY, Term.BAD), T(Variable.PRICE, Term.HIGH)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.FOOD_QUALITY, Term.BAD), T(Variable.ENVIRONMENT, Term.BAD)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.FOOD_QUALITY, Term.BAD), T(Variable.ENVIRONMENT, Term.NORMAL)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.FOOD_QUALITY, Term.BAD), T(Variable.ENVIRONMENT, Term.GOOD)], THEN=T(Variable.RATING, Term.NORMAL)),

    Rule(IF=[T(Variable.FOOD_QUALITY, Term.NORMAL), T(Variable.PRICE, Term.LOW)], THEN=T(Variable.RATING, Term.GOOD)),
    Rule(IF=[T(Variable.FOOD_QUALITY, Term.NORMAL), T(Variable.PRICE, Term.NORMAL)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.FOOD_QUALITY, Term.NORMAL), T(Variable.PRICE, Term.HIGH)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.FOOD_QUALITY, Term.NORMAL), T(Variable.ENVIRONMENT, Term.BAD)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.FOOD_QUALITY, Term.NORMAL), T(Variable.ENVIRONMENT, Term.NORMAL)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.FOOD_QUALITY, Term.NORMAL), T(Variable.ENVIRONMENT, Term.GOOD)], THEN=T(Variable.RATING, Term.GOOD)),

    Rule(IF=[T(Variable.FOOD_QUALITY, Term.GOOD), T(Variable.PRICE, Term.LOW)], THEN=T(Variable.RATING, Term.GOOD)),
    Rule(IF=[T(Variable.FOOD_QUALITY, Term.GOOD), T(Variable.PRICE, Term.NORMAL)], THEN=T(Variable.RATING, Term.GOOD)),
    Rule(IF=[T(Variable.FOOD_QUALITY, Term.GOOD), T(Variable.PRICE, Term.HIGH)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.FOOD_QUALITY, Term.GOOD), T(Variable.ENVIRONMENT, Term.BAD)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.FOOD_QUALITY, Term.GOOD), T(Variable.ENVIRONMENT, Term.NORMAL)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.FOOD_QUALITY, Term.GOOD), T(Variable.ENVIRONMENT, Term.GOOD)], THEN=T(Variable.RATING, Term.GOOD)),

    # # Price vs others
    Rule(IF=[T(Variable.PRICE, Term.LOW), T(Variable.ENVIRONMENT, Term.BAD)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.PRICE, Term.LOW), T(Variable.ENVIRONMENT, Term.NORMAL)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.PRICE, Term.LOW), T(Variable.ENVIRONMENT, Term.GOOD)], THEN=T(Variable.RATING, Term.GOOD)),
    Rule(IF=[T(Variable.PRICE, Term.NORMAL), T(Variable.ENVIRONMENT, Term.BAD)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.PRICE, Term.NORMAL), T(Variable.ENVIRONMENT, Term.NORMAL)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.PRICE, Term.NORMAL), T(Variable.ENVIRONMENT, Term.GOOD)], THEN=T(Variable.RATING, Term.NORMAL)),
    Rule(IF=[T(Variable.PRICE, Term.HIGH), T(Variable.ENVIRONMENT, Term.BAD)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.PRICE, Term.HIGH), T(Variable.ENVIRONMENT, Term.NORMAL)], THEN=T(Variable.RATING, Term.BAD)),
    Rule(IF=[T(Variable.PRICE, Term.HIGH), T(Variable.ENVIRONMENT, Term.GOOD)], THEN=T(Variable.RATING, Term.NORMAL)),
]
