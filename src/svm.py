# Wrapper for liblinear
import sys;
sys.path.append('../liblinear-1.94/python')
import liblinear as ll
import liblinearutil as llu
import os.path

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
        if os.path.isfile('svm.model'):
            self.model = llu.load_model('svm.model')
        else:
            self.model = llu.train(self.ys, self.xs, self.train_param)
            llu.save_model('svm.model', self.model)

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
        self.train_param = '-s 4 -B 1 -e 0.1 -q'
        self.pred_param = '-q'