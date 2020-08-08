from __future__ import division
import torch

import stanza.models.common.seq2seq_constant as constant

"""
 Adapted and modified from the OpenNMT project.

 Class for managing the internals of the beam search process.


         hyp1-hyp1---hyp1 -hyp1
                 \             /
         hyp2 \-hyp2 /-hyp2hyp2
                               /      \
         hyp3-hyp3---hyp3 -hyp3
         ========================

 Takes care of beams, back pointers, and scores.
"""


class Beam(object):
    def __init__(self, size, cuda=False):

        self.size = size
        self.done = False

        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.allScores = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(constant.PAD_ID)]
        self.nextYs[0][0] = constant.SOS_ID

        # The copy indices for each time
        self.copy = []

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.nextYs[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk, copy_indices=None):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `copy_indices` - copy indices (K x ctx_len)

        Returns: True if beam search is complete.
        """
        if self.done:
            return True
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)
        else:
            # first step, expand from the first position
            beamLk = wordLk[0]

        flatBeamLk = beamLk.view(-1)

        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)
        self.allScores.append(self.scores)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        # bestScoreId is the integer ids, and numWords is the integer length.
        # Need to do integer division
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append(bestScoresId - prevK * numWords)
        if copy_indices is not None:
            self.copy.append(copy_indices.index_select(0, prevK))

        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == constant.EOS_ID:
            self.done = True
            self.allScores.append(self.scores)

        return self.done

    def sort_best(self):
        return torch.sort(self.scores, 0, True)

    def get_best(self):
        "Get the score of the best in the beam."
        scores, ids = self.sortBest()
        return scores[1], ids[1]

    def get_hyp(self, k):
        """
        Walk back to construct the full hypothesis.

        Parameters:

             * `k` - the position in the beam to construct.

         Returns: The hypothesis
        """
        hyp = []
        cpy = []
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j+1][k])
            if len(self.copy) > 0:
                cpy.append(self.copy[j][k])
            k = self.prevKs[j][k]
         
        hyp = hyp[::-1]
        cpy = cpy[::-1]
        # postprocess: if cpy index is not -1, use cpy index instead of hyp word
        for i,cidx in enumerate(cpy):
            if cidx >= 0:
                hyp[i] = -(cidx+1) # make index 1-based and flip it for token generation

        return hyp
