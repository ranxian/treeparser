from __future__ import division
import corpus_reader as reader
import os

IDX_IDX     = 0
IDX_TOKEN   = 1
IDX_POS     = 2
IDX_HEAD    = 3

class Parser:

    def __init__(self):
        self.result_path = '../result/predict.conll08'
        self.START_NODE2 = [-2, 'START2', 'START2_POS', '-'] 
        self.START_NODE1 = [-1, 'START1', 'START1_POS', '-']
        self.END_NODE1   = [100, 'END1', 'END1_POS', '-']
        self.END_NODE2   = [101, 'END2', 'END2_POS', '-']

    def train(self, sents):
        # Corner situation
        sents.insert(0, self.START_NODE1)
        sents.insert(0, self.START_NODE2)
        sents.append(self.END_NODE1)
        sents.append(self.END_NODE2)

        

    def predict(self, sents, output=False):
        # fd for output file
        outfile = None
        if output:
            outfile = open(self.result_path, 'w')

        if output != None:
            outfile.close()

def eval(goldpath, predictpath):
    # Evaluate
    os.system('java -jar ../conll08-eval.jar ../data/%s ../result/%s' % (goldpath, predictpath))

parser = Parser()
# Train
parser.train(reader.dev_reader.sents)
# Predict
parser.predict(reader.dev_reader.sents, True)
# Eval
# eval('dev.conll08', 'predict.conll08')

