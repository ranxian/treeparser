# Wrapper for liblinear
import sys;
sys.path.append('../liblinear-1.94/python')
import liblinear as ll
import liblinearutil as llu

class SVM:
    def __init__(self):
        self.reset()

    def add_sample(self, x, y):
        features = {}
        for idx in x:
            features[idx] = 1
        self.xs.append(features)
        self.ys.append(y)

    def train(self):
        self.model = llu.train(self.ys, self.xs, self.train_param)

    def predict(self, features):
        x = {}
        for idx in features:
            x[idx] = 1
        p_labels, p_acc, p_vals = llu.predict([], [x], self.model, self.pred_param)
        return p_labels[0], p_acc[1]

    def reset(self):
        self.xs = []
        self.ys = []
        self.model = None
        self.train_param = '-s 0 -c 4 -B 1 -e 1'
        self.pred_param = '-q'