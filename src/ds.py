
class Span:
    def __init__(self, id, st, en):
        self.id = id
        self.st = st
        self.en = en
        self.score = None
        self.g_i = None
        self.distance = -1

    def __len__(self):
        return self.en-self.st+1

class Instance:
    def __init__(self, id, utterances, frame, speakers):
        self.id = id
        self.utterances = utterances
        self.frame = frame
        self.speakers = speakers

        # Filled in at evaluation time.
        self.tags = None



    #
    # def __getitem__(self, idx):
    #     return (self.tokens[idx], self.corefs[idx], \
    #             self.speakers[idx], self.genre)
    #
    # def __repr__(self):
    #     return 'Document containing %d tokens' % len(self.tokens)
    #
    # def __len__(self):
    #     return len(self.tokens)
    #
    # def sents(self):
    #     """ Regroup raw_text into sentences """
    #
    #     # Get sentence boundaries
    #     sent_idx = [idx+1
    #                 for idx, token in enumerate(self.tokens)
    #                 if token in ['.', '?', '!']]
    #
    #     # Regroup (returns list of lists)
    #     return [self.tokens[i1:i2] for i1, i2 in pairwise([0] + sent_idx)]
    #
    # def spans(self):
    #     """ Create Span object for each span """
    #     return [Span(i1=i[0], i2=i[-1], id=idx,
    #                 speaker=self.speaker(i), genre=self.genre)
    #             for idx, i in enumerate(compute_idx_spans(self.sents))]
    #
    # def truncate(self, MAX=50):
    #     """ Randomly truncate the document to up to MAX sentences """
    #     if len(self.sents) > MAX:
    #         i = random.sample(range(MAX, len(self.sents)), 1)[0]
    #         tokens = flatten(self.sents[i-MAX:i])
    #         return self.__class__(c(self.raw_text), tokens,
    #                               c(self.corefs), c(self.speakers),
    #                               c(self.genre), c(self.filename))
    #     return self
    #
    # def speaker(self, i):
    #     """ Compute speaker of a span """
    #     if self.speakers[i[0]] == self.speakers[i[-1]]:
    #         return self.speakers[i[0]]
    #     return None
