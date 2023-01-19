import skfuzzy as fuzz

from src import Variable, Term, sentiment, service, food_quality, price, environmnet, rating

MF = {
    # Sentiment (likert-scale)
    Variable.SENTIMENT: {
        Term.BAD: fuzz.trapmf(sentiment, [-1, 0, 35, 50]),
        Term.NORMAL: fuzz.trapmf(sentiment, [30, 40, 60, 70]),
        Term.GOOD: fuzz.trapmf(sentiment, [50, 65, 100, 101]),
    },
    # Service (likert-scale)
    Variable.SERVICE: {
        Term.BAD: fuzz.trapmf(service, [-1, 0, 30, 50]),
        Term.NORMAL: fuzz.trapmf(service, [30, 40, 60, 70]),
        Term.GOOD: fuzz.trapmf(service, [50, 70, 100, 101]),
    },
    # Food Quality (likert-scale)
    Variable.FOOD_QUALITY: {
        Term.BAD: fuzz.trimf(food_quality, [-1, 0, 50]),
        Term.NORMAL: fuzz.trapmf(food_quality, [30, 40, 60, 70]),
        Term.GOOD: fuzz.trimf(food_quality, [50, 100, 101]),
    },
    # Price in Ringgit Malaysia (RM)
    Variable.PRICE: {
        Term.LOW: fuzz.gaussmf(price, 5, 10),
        Term.NORMAL: fuzz.gaussmf(price, 32.5, 10),
        Term.HIGH: fuzz.gaussmf(price, 60, 10),
    },
    # Environment (likert-scale)
    Variable.ENVIRONMENT: {
        Term.BAD: fuzz.zmf(environmnet, 10, 40),
        Term.NORMAL: fuzz.gaussmf(environmnet, 50, 20),
        Term.GOOD: fuzz.smf(environmnet, 60, 90),
    },
    # Restaurant Rating (likert-scale)
    Variable.RATING: {
        Term.BAD: fuzz.trapmf(rating, [-10, 0, 30, 50]),
        Term.NORMAL: fuzz.trapmf(rating, [30, 40, 60, 70]),
        Term.GOOD: fuzz.trapmf(rating, [50, 70, 100, 110]),
    },
}
