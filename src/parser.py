from __future__ import division
import corpus_reader as reader
import os
import sys
from perceptron import Perceptron
import time
from collections import defaultdict

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
        
        outfile.write('%5d%20s%20s%20s\n' % (self.idx, self.token, self.pos, str(pidx) if pidx >= 0 else '_'))    
        
        for node in self.right:
            node.printSelf(outfile, self.idx, isConll)

class Parser:
    def __init__(self, lw, rw, iteration):
        self.result_path = '../result/predict.conll08'
        self.token_set = []
        self.pos_set = []
        self.perceptron = Perceptron()
        self.perceptron.tag_set = [ACT_SHIFT, ACT_LEFT, ACT_RIGHT]
        self.iteration = iteration

        self.lw = lw
        self.rw = rw
        self.window_names = []
        self.slnodes = []
        self.srnodes = []

        for i in range(-self.lw, 0):
            absi = abs(i)
            self.window_names.append(str(i))
            token = 'START' + str(absi)
            pos = 'START_POS' + str(absi)
            self.slnodes.append(Node(i-100, token, pos))
            self.token_set.append(token)
            self.pos_set.append(pos)

        self.window_names.append('0-')
        self.window_names.append('0+')

        for i in range(1, self.rw+1):
            self.window_names.append(str(i))
            token = 'END'+str(i)
            pos = 'END_POS' + str(i)
            self.srnodes.append(Node(-i, token, pos))
            self.token_set.append(token)
            self.pos_set.append(pos)

    def train(self, reader):
        sents = reader.sents[0:]
        self.token_set += reader.token_set
        self.pos_set += reader.pos_set
        # For each sentence, retrievel features
        start = time.clock()
        for it in range(self.iteration):
            ntotal = 0
            ncorrect = 0
            for sent in sents:
                # Build a nodes array
                nodes = self._init_nodes(sent)
                # Train
                no_construction = True
                i = self.lw

                while len(nodes) > (1+self.rw+self.lw):
                    if i == len(nodes)-self.rw:
                        if no_construction:
                            break
                        no_construction = True
                        i = self.lw
                    else:
                        # Gain feature
                        features = self._get_features(nodes, i)
                        # Pred action
                        action = self._decide_action(sent, nodes, i)
                        pred_action = self.perceptron.predict(features)
                        self.perceptron.update(action, pred_action, features)

                        ntotal += 1
                        if pred_action == action:
                            ncorrect += 1

                        # Apply action
                        i = self._construct(nodes, i, action)
                        
                        if action == ACT_LEFT or action == ACT_RIGHT:
                            no_construction = False
                        if action < 0:
                            raise BaseException, 'Not a valid action'
            print 'iter: #%d, precision: %.2f%% (%d/%d)' % (it, ncorrect/ntotal*100, ncorrect, ntotal)
        self.perceptron.average_weights
        elapsed = time.clock() - start
        print 'perceptron trained (%.2f secs)' % elapsed

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
            i = self.lw
            while len(nodes) > (1+self.lw+self.rw):
                if i == len(nodes)-self.rw:
                    if no_construction:
                        break
                    no_construction = True
                    i = self.lw
                else:
                    # Extract features
                    features = self._get_features(nodes, i)
                    action = self.perceptron.predict(features)
                    # Apply action
                    i = self._construct(nodes, i, action)
                    
                    if action == ACT_LEFT or action == ACT_RIGHT:
                        no_construction = False
            if output:
                for node in nodes2[self.lw:-self.rw]:
                    outfile.write('%d\t%s\t%s\t%s\t%s\t_\t_\t_\t%d\tX\t_\n' % (node.idx, node.token, node.token, 
                                                                       node.pos, node.pos, node.pidx if node.pidx != None else 0))
                outfile.write('\n')

        if output:
            outfile.close()

        print 'predicted (%.2f secs)' % (time.clock() - start)

    def _get_features(self, stack, queue):
        def add(name, *args):
            features['_'.join((name, ) + tuple(args))] = 1

        features = defaultdict(int)

        s0 = stack[-1]
        n0 = queue[0]
        n1 = queue[1]
        n2 = queue[2]

        add('bias')
        # feat_names = ['pos', 'lex', 'chLlex', 'chRlex', 'chLpos', 'chRpos', 'pl', '2pl']
        # Unigram-like feature
        add('0-:wp', s0.token, s0.pos)
        add('0-:w', s0.token)
        add('0-:p', s0.pos)
        add('0+:wp', n0.token, n0.pos)
        add('0+:w', n0.token, n0.pos)
        add('0+:p', n0.token, n0.pos)
        add('1:wp', n1.token, n1.pos)
        add('1:w', n1.token)
        add('1:p', n1.pos)
        add('2:wp', n2.token, n2.pos)
        add('2:w', n2.token)
        add('2:p', n2.pos)
        # Bigram-like feature
        # S0wpN0wp; S0wpN0w; S0wN0wp; S0wpN0p; S0pN0wp; S0wN0w; S0pN0p
        add('0-wp+0+:wp', s0.token, s0.pos, n0.token, n0.pos)
        add('0-wp+0+:w', s0.token, s0.pos, n0.token)
        add('0-w+0+:wp', s0.token, n0.token, n0.pos)
        add('0-wp0+:p', s0.token, s0.pos, n0.pos)
        add('0-p:0+wp', s0.pos, n0.token, n0.pos)
        add('0-w:0+w', s0.token, n0.token)
        add('0-p:0+p', s0.pos, n0.pos)
        # Trigram-like feature
        # N0pN1pN2p; S0pN0pN1p; S0pS0lpN0p; S0pS0rpN0p; S0pN0pN0lp
        add('0+p:1p:2p', n0.pos, n1.pos, n2.pos)
        add('0-p:0+p:1p', s0.pos, n0.pos, n1.pos)
        if len(s0.left) > 0:
            add('0-p:0-lp:0+p', s0.pos, s0.left[0].pos, n0.pos)
        if len(s0.right) > 0:
            add('0-p:0-rp:1p', s0.pos, s0.right[-1].pos, n0.pos)
        if len(n0.left) > 0:
            add('0-p:0+p:0+lp', s0.pos, n0.pos, n0.left[0].pos)

        if len(queue) == 1+self.rw:
            add('last')   

        return features

    def _decide_action(self, sent, nodes, i):
        nodei = nodes[i]
        nodej = nodes[i+1]
        action = ACT_SHIFT
        # See if i->j (Right)
        if nodej.idx > 0 and sent[nodei.idx-1][IDX_HEAD] == nodej.idx:
            # Check no other node is nodei's child
            complete = True
            for node in nodes[self.lw:-self.rw]:
                if sent[node.idx-1][IDX_HEAD] == nodei.idx:
                    complete = False
                    break
            if complete:
                action = ACT_RIGHT
        elif nodei.idx > 0 and sent[nodej.idx-1][IDX_HEAD] == nodei.idx:
            # See if j->i
            # Check no other node is nodej's child
            complete = True
            for node in nodes[self.lw:-self.rw]:
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
        elif action == ACT_LEFT: # j->i
            nodes.remove(nodej)
            nodei.right.append(nodej)
            nodej.pidx = nodei.idx
        elif action == ACT_RIGHT: # i->j
            nodes.remove(nodei)
            nodej.left.append(nodei)
            nodei.pidx = nodej.idx
        else:
            raise BaseException, 'Not a valid action'

        return i

    def _init_nodes(self, sent):
        nodes = []
        # Build a nodes array
        for node in self.slnodes:
            nodes.append(node)
        for word in sent:
            nodes.append(Node(word[IDX_IDX], word[IDX_TOKEN], word[IDX_POS]))
        for node in self.srnodes:
            nodes.append(node)

        return nodes
        

def eval(goldpath, predictpath):
    # Evaluate
    os.system('java -jar ../conll08-eval.jar ../data/%s ../result/%s' % (goldpath, predictpath))

niter = int(sys.argv[1])

parser = Parser(1, 3, niter)
# Train
parser.train(reader.trn_reader)
# Predict
parser.predict(reader.dev_reader.sents[0:], True)
# Eval
eval('dev.conll08', 'predict.conll08')

logfile.close()
