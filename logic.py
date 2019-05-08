import numpy as np
from keras.models import load_model

STOP_MODEL_PATH = "stop_not_stop.model"
YIELD_MODEL_PATH = "yield_not_yield.model"
RAIL_MODEL_PATH = "railroad_not_railroad.model"
LIGHT_MODEL_PATH = "light_not_light.model"
SPEED_MODEL_PATH = "speed_not_speed.model"

class Logic:
    def __init__(self):
        self.stop_model = load_model(STOP_MODEL_PATH)
        self.yield_model = load_model(YIELD_MODEL_PATH)
        self.rail_model = load_model(RAIL_MODEL_PATH)
        self.light_model = load_model(LIGHT_MODEL_PATH)
        self.speed_model = load_model(SPEED_MODEL_PATH)
        self.consecutive_stop = 0

    def is_stop(self, image):
        other, stop = self.stop_model.predict(image)[0]
        return [stop > other, max([other, stop])]

    def is_yield(self, image):
        other, _yield = self.yield_model.predict(image)[0]
        return [_yield > other, max([other, _yield])]

    def is_rail(self, image):
        other, rail = self.rail_model.predict(image)[0]
        return [rail > other, max([other, rail])]

    def is_light(self, image):
        other, light = self.light_model.predict(image)[0]
        return [light > other, max([other, light])]

    def is_speed(self, image):
        other, speed = self.speed_model.predict(image)[0]
        return [speed > other, max([other, speed])]

    def identify(self, image):
        rail_result = self.is_rail(image)
        if rail_result[0]:
            return ["rails", rail_result[1]]

        stop_result = self.is_stop(image)
        if stop_result[0]:
            return ["stop", stop_result[1]]

        yield_result = self.is_yield(image)
        if yield_result[0]:
            return ["yield", yield_result[1]]

        speed_result = self.is_speed(image)
        if speed_result[0]:
            return ["speed", speed_result[1]]

        light_result = self.is_light(image)
        if light_result[0]:
            return ["light", light_result[1]]

        return ["nothing", max([yield_result[1], stop_result[1], rail_result[1], light_result[1], speed_result[1]])]

    def action_to_take(self, image):
        action = "move"
        sign, confidence = self.identify(image)
        if (sign == "stop" or sign == "rails") and self.consecutive_stop < 10:
            self.consecutive_stop += 1
            action = "stop"
        elif (sign == "stop" or sign == "rails") and self.consecutive_stop >= 10:
            # means we are reading the same stop sign
            self.consecutive_stop += 1
            action == "move"
            sign = "same " + sign
        elif sign == "yield":
            action = "slow"
            self.consecutive_stop = 0
        elif sign == "nothing":
            sign = None
            self.consecutive_stop = 0
        else:
            self.consecutive_stop = 0
        return [action, sign, confidence]
