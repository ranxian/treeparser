class CorpusReader:
    def __init__(self, filename):
        f = open(filename, 'r')

        self.sents = []
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
                sp2 = [int(sp[0]), sp[1], sp[3]]
                for t in sp[5:]:
                    sp2.append(t)
                sent.append(sp2)

        f.close()
        print len(self.sents)

dev_data_path = '../data/dev.conll08'
trn_data_path = '../data/trn.conll08'
# tst_data_path = '../data/tst.conll08'

dev_reader = CorpusReader(dev_data_path)
trn_reader = CorpusReader(trn_data_path)
# tst_sents = CorpusReader(tst_data_path)

if __name__ == '__main__':
    reader = CorpusReader('../data/dev.conll08')