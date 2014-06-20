class CorpusReader:
    def __init__(self, filename):
        f = open(filename, 'r')

        self.sents = []
        self.pos_set = []
        self.token_set = []
        sent = []

        while True:
            line = f.readline()
            if line == '':
                break

            line = line.rstrip('\n')

            if line == '':
                self.sents.append(sent)
                sent = []
            else:
                sp = line.split('\t')
                sp2 = [int(sp[0]), sp[1], sp[3], int(sp[8])]
                self.token_set.append(sp[1])
                self.pos_set.append(sp[3])
                sent.append(sp2)
        f.close()
        
        print 'nSample: %d' % len(self.token_set),
        self.token_set = list(set(self.token_set))
        print 'Vocabulary: %d' % len(self.token_set),
        self.pos_set = list(set(self.pos_set))
        print 'Pos %d' % (len(self.pos_set)),
        print 'Sents %d' % len(self.sents)

dev_data_path = '../data/dev.conll08'
trn_data_path = '../data/trn.conll08'
# tst_data_path = '../data/tst.conll08'

dev_reader = CorpusReader(dev_data_path)
trn_reader = CorpusReader(trn_data_path)
# tst_sents = CorpusReader(tst_data_path)

if __name__ == '__main__':
    reader = CorpusReader('../data/dev.conll08')