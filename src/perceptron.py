from __future__ import division
from collections import defaultdict

class Perceptron:
    def __init__(self):
        self.reset()

    def get_score(self, features, tag):
        score = 0
        for feature, value in features.items():
            if feature not in self.weights or value == 0:
                continue
            weights = self.weights[feature]
            score += weights[tag] * value
        return score

    def predict(self, features):
        scores = defaultdict(float)
        for feature, value in features.items():
            if feature not in self.weights or value == 0:
                continue
            weights = self.weights[feature]
            for tag, weight in weights.items():
                scores[tag] += value * weight
        return max(self.tag_set, key=lambda tag: scores[tag]), scores

    def update(self, tag, pred, features):
        def update_feature(tag, feature, value):
            key = (feature, tag)
            self._totals[key] += (self._time - self._timestamps[key]) * self.weights[feature][tag]
            self._timestamps[key] = self._time
            self.weights[feature][tag] += value

        self._time += 1
        if tag != pred:
            for feature in features:
                update_feature(tag, feature, 1.0)
                update_feature(pred, feature, -1.0)

    def average_weights(self):
        for feature, weights in self.weights.items():
            new_weights = defaultdict(float)
            for tag, weight in weights.items():
                key = (feature, tag)
                total = self._totals[key] + (self._time - self._timestamps[key]) * weight
                aver = total / self._time
                new_weights[tag] = aver
            self.weights[feature] = new_weights

    def reset(self):
        self.weights = defaultdict(lambda: defaultdict(float))
        self.tag_set = set()
        self._totals = defaultdict(int)
        self._timestamps = defaultdict(int)
        self._time = 0
