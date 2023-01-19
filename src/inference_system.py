from functools import reduce

import numpy as np
import skfuzzy as fuzz
from matplotlib import pyplot as plt
import streamlit as st

from src import rules, Variable, MF, sys_variables, rating


class RestaurantRatingSystem:

    def __init__(self) -> None:
        self.rules = rules

    def predict(self, sentiment, service, food_quality, price, environment):

        inputs = {
            Variable.SENTIMENT: sentiment,
            Variable.SERVICE: service,
            Variable.FOOD_QUALITY: food_quality,
            Variable.PRICE: price,
            Variable.ENVIRONMENT: environment,
        }

        self.fuzzy_output = reduce(np.fmax, [rule.predict(inputs) for rule in self.rules])


        return fuzz.defuzz(rating, self.fuzzy_output, 'centroid')

    def visualize(self):
        fig, ax = plt.subplots(figsize=(15, 10))
        var = Variable.RATING

        for item, mf in MF[var].items():
            ax.plot(mf)

        ax.set_title(var.name)
        ax.legend(MF[var].keys(), loc='upper right')
        ax.plot(self.fuzzy_output)
        ax.fill_between(sys_variables[var.value], self.fuzzy_output)

        st.pyplot(fig)
