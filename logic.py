import numpy as np
from keras.models import load_model

STOP_MODEL_PATH = "stop_not_stop.model"
YIELD_MODEL_PATH = "yield_not_yield.model"
RAIL_MODEL_PATH = "railroad_not_railroad.model"

class Logic:
    def __init__(self):
        self.stop_model = load_model(STOP_MODEL_PATH)
        self.yield_model = load_model(YIELD_MODEL_PATH)
        self.rail_model = load_model(RAIL_MODEL_PATH)

    def is_stop(self, image):
        other, stop = self.stop_model.predict(image)[0]
        return [stop > other, max([other, stop])]

    def is_yield(self, image):
        other, _yield = self.yield_model.predict(image)[0]
        return [_yield > other, max([other, _yield])]

    def is_rail(self, image):
        other, rail = self.rail_model.predict(image)[0]
        return [rail > other, max([other, rail])]

    def identify(self, image):
        stop_result = self.is_stop(image)
        rail_result = self.is_rail(image)
        if stop_result[0] or rail_result[0]:
            return ["stop", max([stop_result[1], rail_result[1]])]

        yield_result = self.is_yield(image)
        if yield_result[0]:
            return ["yield", yield_result[1]]

        return ["nothing", max([yield_result[1], stop_result[1], rail_result[1]])]

    def action_to_take(self, image):
        action = "move"
        sign, confidence = self.identify(image)
        if sign == "stop":
            action = "stop"
        elif sign == "yield":
            action = "slow"
        return [action, confidence]
