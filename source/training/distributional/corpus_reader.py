import codecs


class CorpusReader(object):
    def __init__(self, corpus_file):
        self.corpus_file = corpus_file

    def __iter__(self):
        # Add the unknown word
        yield ['<unk>'] * 10

        with codecs.open(self.corpus_file, 'r', 'utf-8') as f_in:
            for line in f_in:
                yield line.strip().lower().split()
