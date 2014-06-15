from __future__ import division
import corpus_reader as reader
import os
from svm import SVM

IDX_IDX     = 0
IDX_TOKEN   = 1
IDX_POS     = 2
IDX_HEAD    = 3

ACT_SHIFT   = 0
ACT_LEFT    = 1
ACT_RIGHT   = 2

class Node:
    def __init__(self, idx, token, pos):
        self.left = []
        self.right = []
        self.idx = idx
        self.token = token
        self.pos = pos

    def __str__(self):
        return 'hello'

    def printSelf(self, pidx):
        for node in self.left:
            node.printSelf(self.idx)
        print '%5d%20s%20s%20s' % (self.idx, self.token, self.pos, str(pidx) if pidx >= 0 else '-')
        for node in self.right:
            node.printSelf(self.idx)

class Parser:

    def __init__(self):
        self.result_path = '../result/predict.conll08'
        self.START_NODE2 = Node(-1, 'START2', 'START2_POS')
        self.START_NODE1 = Node(-2, 'START1', 'START1_POS')
        self.END_NODE1   = Node(-3, 'END1', 'END1_POS')
        self.END_NODE2   = Node(-4, 'END2', 'END2_POS')
        self.svmLS = SVM()
        self.svmRS = SVM()
        self.svmLR = SVM()

    def train(self, sents):
        # For each sentence, retrievel features
        for sent in sents:
            nodes = []
            # Build a nodes array
            nodes.append(self.START_NODE1)
            nodes.append(self.START_NODE2)
            for word in sent:
                nodes.append(self._build_node(word))
            nodes.append(self.END_NODE1)
            nodes.append(self.END_NODE2)
            # Train
            no_construction = True
            i = 2
            cnt = 0
            while len(nodes) > 5:
                if i == len(nodes)-2:
                    if no_construction:
                        break
                    i = 2
                else:
                    # Gain feature
                    features = self._get_features(nodes, i)
                    # Pred action
                    action = self._decide_action(nodes, i)
                    # Apply action
                    action = 1+(cnt%2)
                    cnt += 1
                    i = self._construct(nodes, i, action)
                    
                    if action == ACT_LEFT or action == ACT_RIGHT:
                        no_construction = False
                    # Add sample to svm    
                    if action == ACT_LEFT:
                        self.svmLR.add_sample(features, +1)
                        self.svmLS.add_sample(features, +1)
                    elif action == ACT_RIGHT:
                        self.svmLR.add_sample(features, -1)
                        self.svmRS.add_sample(features, +1)
                    elif action == ACT_SHIFT:
                        self.svmLS.add_sample(features, -1)
                        self.svmRS.add_sample(features, -1)
                    else:
                        raise BaseException, 'No a valid action'
            for node in nodes[2:-2]:
                node.printSelf(-1)
        self.svmLR.train()
        self.svmRS.train()
        self.svmLS.train()

    def predict(self, sents, output=False):
        # fd for output file
        outfile = None
        if output:
            outfile = open(self.result_path, 'w')

        if output != None:
            outfile.close()

    def _build_node(self, word):
        # Build node from a sentence in corpus
        node = Node(word[IDX_IDX], word[IDX_TOKEN], word[IDX_POS])
        return node

    def _get_features(self, nodes, i):
        pass

    def _decide_action(self, nodes, i):
        pass

    def _construct(self, nodes, i, action):
        nodei = nodes[i]
        nodej = nodes[i+1]
        if action == ACT_SHIFT:
            i += 1
        elif action == ACT_LEFT:
            nodes.remove(nodej)
            nodei.right.append(nodej)
        elif action == ACT_RIGHT:
            nodes.remove(nodei)
            nodej.left.append(nodei)
        else:
            raise BaseException, 'Not a valid action'

        return i
        

def eval(goldpath, predictpath):
    # Evaluate
    os.system('java -jar ../conll08-eval.jar ../data/%s ../result/%s' % (goldpath, predictpath))

parser = Parser()
# Train
parser.train(reader.dev_reader.sents[0:1])
# Predict
parser.predict(reader.dev_reader.sents[0:1], True)
# Eval
# eval('dev.conll08', 'predict.conll08')

