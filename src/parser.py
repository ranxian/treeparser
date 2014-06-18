from __future__ import division
import corpus_reader as reader
import os
from svm import SVM
import time

IDX_IDX     = 0
IDX_TOKEN   = 1
IDX_POS     = 2
IDX_HEAD    = 3

ACT_SHIFT   = 0
ACT_LEFT    = 1
ACT_RIGHT   = 2

logfile = open('log.txt', 'w')

class Node:
    def __init__(self, idx, token, pos):
        self.left = []
        self.right = []
        self.idx = idx
        self.token = token
        self.pos = pos
        self.pidx = None

    def printSelf(self, outfile, pidx, isConll=False):
        for node in self.left:
            node.printSelf(outfile, self.idx, isConll)

        if isConll:
            outfile.write(('%d\t%s\t%s\t%s\t%s\t_\t_\t_\t%d\tX\t_\n') % (self.idx, self.token, self.token, self.pos, self.pos, pidx))
        else:
            outfile.write('%5d%20s%20s%20s\n' % (self.idx, self.token, self.pos, str(pidx) if pidx >= 0 else '_'))    
        
        for node in self.right:
            node.printSelf(outfile, self.idx, isConll)

class Parser:
    def __init__(self):
        self.result_path = '../result/predict.conll08'
        self.token_set = []
        self.pos_set = []
        self.svm = SVM()

        self.lw = 2
        self.rw = 4
        self.window_names = []
        self.slnodes = []
        self.srnodes = []

        for i in range(-self.lw, 0):
            absi = abs(i)
            self.window_names.append(str(i))
            self.slnodes.append(Node(i-100, 'START' + str(absi), 'START_POS' + str(absi)))
        self.window_names.append('0-')
        self.window_names.append('0+')
        for i in range(1, self.rw+1):
            self.window_names.append(str(i))
            self.srnodes.append(Node(-i, 'END'+str(i), 'END_POS' + str(i)))

    def train(self, reader):
        sents = reader.sents[0:]
        self.token_set = reader.token_set
        self.pos_set = reader.pos_set
        self._init_feature_map()
        # For each sentence, retrievel features
        start = time.clock()
        for sent in sents:
            # Build a nodes array
            nodes = self._init_nodes(sent)
            # Train
            no_construction = True
            i = 2
            while len(nodes) > 5:
                if i == len(nodes)-2:
                    if no_construction:
                        break
                    no_construction = True
                    i = 2
                else:
                    # Gain feature
                    features = self._get_features(nodes, i)
                    # Pred action
                    action = self._decide_action(sent, nodes, i)
                    # Apply action
                    i = self._construct(nodes, i, action)
                    
                    if action == ACT_LEFT or action == ACT_RIGHT:
                        no_construction = False
                    if action < 0:
                        raise BaseException, 'Not a valid action'
                    self.svm.add_sample(features, action)
        elapsed = time.clock() - start
        print 'features added (%.2f secs)' % elapsed
        print 'Training SVM'
        start = time.clock()
        self.svm.train()
        elapsed = time.clock() - start
        print 'Trained (%.2f secs)' % elapsed

    def predict(self, sents, output=False):
        # fd for output file
        start = time.clock()
        outfile = None
        if output:
            outfile = open(self.result_path, 'w')

        for sent in sents:
            nodes = self._init_nodes(sent)
            nodes2 = [node for node in nodes]
            no_construction = True
            i = 2
            while len(nodes) > 7:
                if i == len(nodes)-4:
                    if no_construction:
                        break
                    no_construction = True
                    i = 2
                else:
                    # Gain feature
                    features = self._get_features(nodes, i)
                    # Pred action
                    action = self.svm.predict(features)[0]
                    # Apply action
                    i = self._construct(nodes, i, action)
                    
                    if action == ACT_LEFT or action == ACT_RIGHT:
                        no_construction = False
            if output:
                for node in nodes2[2:-4]:
                    outfile.write('%d\t%s\t%s\t%s\t%s\t_\t_\t_\t%d\tX\t_\n' % (node.idx, node.token, node.token, 
                                                                       node.pos, node.pos, node.pidx if node.pidx != None else 0))
                outfile.write('\n')

        if output:
            outfile.close()
        print 'predicted (%.2f secs)' % (time.clock() - start)

    def _build_node(self, word):
        # Build node from a sentence in corpus
        node = Node(word[IDX_IDX], word[IDX_TOKEN], word[IDX_POS])
        return node

    def _get_features(self, nodes, i):
        features = []
        def add_feature(position, name, value):
            feat = position + ':' + name + ':' + value
            featidx = self.feature_map.get(feat)
            if featidx != None:
                features.append(featidx)
            else:
                pass

        window_names = ['-2', '-1', '0-', '0+', '1', '2', '3', '4']
        idx = 0
        for node in nodes[i-2:i+5]:
            position = window_names[idx]
            add_feature(position, 'pos', node.pos)
            add_feature(position, 'lex', node.token)
            if len(node.left) > 0:
                add_feature(position, 'chLlex', node.left[0].token)
            if len(node.right) > 0:
                add_feature(position, 'chRlex', node.right[0].token)
            idx += 1

        return features

    def _decide_action(self, sent, nodes, i):
        nodei = nodes[i]
        nodej = nodes[i+1]
        action = ACT_SHIFT
        # See if i->j (Right)
        if nodei.idx > 0 and sent[nodei.idx-1][IDX_HEAD] == nodej.idx:
            # Check no other node is nodei's child
            complete = True
            for node in nodes[2:-4]:
                if sent[node.idx-1][IDX_HEAD] == nodei.idx:
                    complete = False
                    break
            if complete:
                action = ACT_RIGHT
        elif nodej.idx > 0 and sent[nodej.idx-1][IDX_HEAD] == nodei.idx:
            # Check no other node is nodej's child
            complete = True
            for node in nodes[2:-4]:
                if sent[node.idx-1][IDX_HEAD] == nodej.idx:
                    complete = False
                    break
            if complete:
                action = ACT_LEFT

        # print 'action is %d' % action
        return action

    def _construct(self, nodes, i, action):
        nodei = nodes[i]
        nodej = nodes[i+1]
        if action == ACT_SHIFT:
            i += 1
        elif action == ACT_LEFT:
            nodes.remove(nodej)
            nodei.right.append(nodej)
            nodej.pidx = nodei.idx
        elif action == ACT_RIGHT:
            nodes.remove(nodei)
            nodej.left.append(nodei)
            nodei.pidx = nodej.idx
        else:
            raise BaseException, 'Not a valid action'

        return i

    def _init_nodes(self, sent):
        # self.START_NODE4 = Node(-1, 'START4', 'START4_POS')
        # self.START_NODE3 = Node(-2, 'START3', 'START3_POS')
        self.START_NODE2 = Node(-3, 'START2', 'START2_POS')
        self.START_NODE1 = Node(-4, 'START1', 'START1_POS')
        self.END_NODE1   = Node(-5, 'END1', 'END1_POS')
        self.END_NODE2   = Node(-6, 'END2', 'END2_POS')
        self.END_NODE3   = Node(-7, 'END3', 'END3_POS')
        self.END_NODE4   = Node(-8, 'END4', 'END4_POS')

        nodes = []
        # Build a nodes array
        # nodes.append(self.START_NODE4)
        # nodes.append(self.START_NODE3)
        nodes.append(self.START_NODE2)
        nodes.append(self.START_NODE1)
        for word in sent:
            nodes.append(self._build_node(word))
        nodes.append(self.END_NODE1)
        nodes.append(self.END_NODE2)
        nodes.append(self.END_NODE3)
        nodes.append(self.END_NODE4)

        return nodes

    def _init_feature_map(self):
        # -2:pos:NN => 0
        self.token_set += ['START1', 'START2', 'END1', 'END2', 'END3', 'END4']
        self.pos_set += ['START1_POS', 'START2_POS', 'END1_POS', 'END2_POS', 'END3_POS', 'END4_POS']
        window_names = ['-2', '-1', '0-', '0+', '1', '2', '3', '4']
        feat_names = ['pos', 'lex', 'chLlex', 'chRlex']
        self.feature_map = {}

        cnt = 0
        for windname in window_names:
            for featname in feat_names:
                if featname.endswith('pos'):
                    for pos in self.pos_set:
                        key = windname + ':' + featname + ':' + pos
                        self.feature_map[key] = cnt
                        cnt += 1
                elif featname.endswith('lex'):
                    for lex in self.token_set:
                        key = windname + ':' + featname + ':' + lex
                        self.feature_map[key] = cnt
                        cnt += 1

        print len(self.feature_map)
        

def eval(goldpath, predictpath):
    # Evaluate
    os.system('java -jar ../conll08-eval.jar ../data/%s ../result/%s' % (goldpath, predictpath))

parser = Parser()
# Train
parser.train(reader.trn_reader)
# Predict
parser.predict(reader.dev_reader.sents[0:], True)
# Eval
# eval('dev.conll08', 'predict.conll08')

logfile.close()
