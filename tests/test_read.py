"""
Tests to read a stored protobuf.
Also serves as an example of how to parse sentences, tokens, pos, lemma,
ner, dependencies and mentions.

The test corresponds to annotations for the following sentence:
    Chris wrote a simple sentence that he parsed with Stanford CoreNLP.
"""

import os
import unittest
from corenlp_protobuf import Document, Sentence, Token, DependencyGraph, CorefChain
from corenlp_protobuf import parseFromDelimitedString, to_text

class TestProtobuf(unittest.TestCase):
    def setUp(self):
        self.text = "Chris wrote a simple sentence that he parsed with Stanford CoreNLP.\n"
        test_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test.dat')
        with open(test_data, 'rb') as f:
            self.buf = f.read()

        self.doc = Document()
        parseFromDelimitedString(self.doc, self.buf)

    def test_parse_protobuf(self):
        self.assertEqual(4239, self.doc.ByteSize(), "Could not read input correctly")

    def test_document_text(self):
        self.assertEqual(self.text, self.doc.text)

    def test_sentences(self):
        self.assertEqual(1, len(self.doc.sentence))

        sentence = self.doc.sentence[0]
        self.assertTrue(isinstance(sentence, Sentence))
        self.assertEqual(67, sentence.characterOffsetEnd - sentence.characterOffsetBegin) # Sentence length
        self.assertEqual('', sentence.text) # Note that the sentence text should actually be recovered from the tokens.
        self.assertEqual(self.text[:-1], to_text(sentence)) # Note that the sentence text should actually be recovered from the tokens.

    def test_tokens(self):
        sentence = self.doc.sentence[0]
        tokens = sentence.token
        self.assertEqual(12, len(tokens))
        self.assertTrue(isinstance(tokens[0], Token))

        # Word
        words = "Chris wrote a simple sentence that he parsed with Stanford CoreNLP .".split()
        words_ = [t.word for t in tokens]
        self.assertEqual(words, words_)

        # Lemma
        lemmas = "Chris write a simple sentence that he parse with Stanford CoreNLP .".split()
        lemmas_ = [t.lemma for t in tokens]
        self.assertEqual(lemmas, lemmas_)

        # POS
        pos = "NNP VBD DT JJ NN IN PRP VBD IN NNP NNP .".split()
        pos_ = [t.pos for t in tokens]
        self.assertEqual(pos, pos_)

        # NER
        ner = "PERSON O O O O O O O O ORGANIZATION O O".split()
        ner_ = [t.ner for t in tokens]
        self.assertEqual(ner, ner_)

        # character offsets
        begin = [int(i) for i in "0 6 12 14 21 30 35 38 45 50 59 66".split()]
        end =   [int(i) for i in "5 11 13 20 29 34 37 44 49 58 66 67".split()]
        begin_ = [t.beginChar for t in tokens]
        end_ = [t.endChar for t in tokens]
        self.assertEqual(begin, begin_)
        self.assertEqual(end, end_)

    def test_dependency_parse(self):
        """
        Extract the dependency parse from the annotation.
        """
        sentence = self.doc.sentence[0]

        # You can choose from the following types of dependencies.
        # In general, you'll want enhancedPlusPlus
        self.assertTrue(sentence.basicDependencies.ByteSize() > 0)
        self.assertTrue(sentence.enhancedDependencies.ByteSize() > 0)
        self.assertTrue(sentence.enhancedPlusPlusDependencies.ByteSize() > 0)

        tree = sentence.enhancedPlusPlusDependencies
        self.assertTrue(isinstance(tree, DependencyGraph))
        # Indices are 1-indexd with 0 being the "pseudo root"
        self.assertEqual([2], tree.root) # 'wrote' is the root.
        # There are as many nodes as there are tokens.
        self.assertEqual(len(sentence.token), len(tree.node))

        # Enhanced++ depdencies often contain additional edges and are
        # not trees -- here, 'parsed' would also have an edge to
        # 'sentence'
        self.assertEqual(12, len(tree.edge))

        # This edge goes from "wrote" to "Chirs"
        edge = tree.edge[0]
        self.assertEqual(2, edge.source)
        self.assertEqual(1, edge.target)
        self.assertEqual("nsubj", edge.dep)

    def test_coref_chain(self):
        """
        Extract the corefence chains from the annotation.
        """
        # Coreference chains span sentences and are stored in the
        # document.
        chains = self.doc.corefChain

        # In this document there is 1 chain with Chris and he.
        self.assertEqual(1, len(chains))
        chain = chains[0]
        self.assertTrue(isinstance(chain, CorefChain))
        self.assertEqual(0, chain.mention[0].beginIndex) # Starts at token 0 == 'Chris'
        self.assertEqual(1, chain.mention[0].endIndex)
        self.assertEqual("MALE", chain.mention[0].gender)

        self.assertEqual(6, chain.mention[1].beginIndex) # Starts at token 6 == 'he'
        self.assertEqual(7, chain.mention[1].endIndex)
        self.assertEqual("MALE", chain.mention[1].gender)

        self.assertEqual(0, chain.representative) # The head of the chain is 'Chris'


if __name__ == "__main__":
    unittest.main()
