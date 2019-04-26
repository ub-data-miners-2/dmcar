import numpy as np
from keras.models import load_model

MODEL_PATH = "traffic_sign.model"
DICT = ["other", "speed", "stop", "yield"]

class Logic:
    def __init__(self):
        self.model = load_model(MODEL_PATH)

    def identify(self, image):
        predictions = self.model.predict(image)[0]
        answer = max(predictions)
        index = np.where(predictions == answer)
        return [ DICT[index[0][0]], answer]

    def action_to_take(self, image):
        action = "move"
        sign, confidence = self.identify(image)
        if sign == "stop":
            action = "stop"
        elif sign == "yield":
            action = "slow"
        return [action, confidence]
