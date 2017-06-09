import six
from abc import abstractmethod

if six.PY2:
    from itertools import izip
else:
    izip = zip

import requests
from collections import defaultdict
from google.protobuf.internal.decoder import _DecodeVarint
from stanza.text import to_unicode

from . import CoreNLP_pb2
from stanza.nlp.data import Document, Sentence, Token, Entity

__author__ = 'kelvinguu, vzhong, wmonroe4'


class AnnotationException(Exception):
    """
    Exception raised when there was an error communicating with the CoreNLP server.
    """
    pass

class TimeoutException(AnnotationException):
    """
    Exception raised when the CoreNLP server timed out.
    """
    pass


class CoreNLPClient(object):
    """
    A CoreNLP client to the Stanford CoreNLP server.
    """

    DEFAULT_ANNOTATORS = "tokenize ssplit lemma pos ner depparse".split()

    def __init__(self, server='http://localhost:9000', default_annotators=DEFAULT_ANNOTATORS):
        """
        Constructor.
        :param (str) server: url of the CoreNLP server.
        """
        self.server = server
        self.default_annotators = default_annotators
        assert requests.get(self.server).ok, 'Stanford CoreNLP server was not found at location {}'.format(self.server)

    def _request(self, text, properties):
        """Send a request to the CoreNLP server.

        :param (str | unicode) text: raw text for the CoreNLPServer to parse
        :param (dict) properties: properties that the server expects
        :return: request result
        """
        text = to_unicode(text)  # ensures unicode
        try:
            r = requests.post(self.server, params={'properties': str(properties)}, data=text.encode('utf-8'))
            r.raise_for_status()
            return r
        except requests.HTTPError:
            if r.text == "CoreNLP request timed out. Your document may be too long.":
                raise TimeoutException(r.text)
            else:
                raise AnnotationException(r.text)

    def annotate_json(self, text, annotators=None):
        """Return a JSON dict from the CoreNLP server, containing annotations of the text.

        :param (str) text: Text to annotate.
        :param (list[str]) annotators: a list of annotator names

        :return (dict): a dict of annotations
        """
        properties = {
            'annotators': ','.join(annotators or self.default_annotators),
            'outputFormat': 'json',
        }
        return self._request(text, properties).json(strict=False)

    def annotate_proto(self, text, annotators=None):
        """Return a Document protocol buffer from the CoreNLP server, containing annotations of the text.

        :param (str) text: text to be annotated
        :param (list[str]) annotators: a list of annotator names

        :return (CoreNLP_pb2.Document): a Document protocol buffer
        """
        properties = {
            'annotators': ','.join(annotators or self.default_annotators),
            'outputFormat': 'serialized',
            'serializer': 'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer'
        }
        r = self._request(text, properties)
        buffer = r.content  # bytes

        size, pos = _DecodeVarint(buffer, 0)
        buffer = buffer[pos:(pos + size)]
        doc = CoreNLP_pb2.Document()
        doc.ParseFromString(buffer)
        return doc

    def annotate(self, text, annotators=None):
        """Return an AnnotatedDocument from the CoreNLP server.

        :param (str) text: text to be annotated
        :param (list[str]) annotators: a list of annotator names

        See a list of valid annotator names here:
          http://stanfordnlp.github.io/CoreNLP/annotators.html

        :return (AnnotatedDocument): an annotated document
        """
        doc_pb = self.annotate_proto(text, annotators)
        return AnnotatedDocument.from_pb(doc_pb)

class ProtobufBacked(object):
    """An object backed by a Protocol buffer.

    ProtobufBacked objects should keep their constructors private.
    They should be exclusively initialized using `from_pb`.
    """
    @classmethod
    def from_pb(cls, pb):
        """Instantiate the object from a protocol buffer.

        Args:
            pb (protobuf)

        Save a reference to the protocol buffer on the object.
        """
        obj = cls._from_pb(pb)
        obj._pb = pb
        return obj

    @abstractmethod
    def _from_pb(cls, pb):
        """Instantiate the object from a protocol buffer.

        Note: this should be a classmethod.
        """
        pass

    @property
    def pb(self):
        """Get the backing protocol buffer."""
        return self._pb

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.pb == other.pb

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def json(self):
        """The object represented as JSON.

        In the future, this should be computed from the protocol buffer. For now, it is manually set.
        """
        try:
            return self._json
        except AttributeError:
            raise AttributeError('No JSON representation available.')

    @json.setter
    def json(self, json_dict):
        self._json = json_dict

    def to_json(self):
        """Same as the json property.

        Provided just because people are accustomed to calling `to_json` on objects.
        """
        return self.json

    @abstractmethod
    def json_to_pb(cls, json_dict):
        """Convert JSON to protocol buffer.

        Note: This should be a classmethod.
        """
        pass

    @classmethod
    def from_json(cls, json_dict):
        pb = cls.json_to_pb(json_dict)
        obj = cls.from_pb(pb)
        obj.json = json_dict  # set the JSON
        return obj

class AnnotatedDocument(Document, ProtobufBacked):
    """
    A shim over the protobuffer exposing key methods.
    """
    @classmethod
    def _from_pb(cls, pb):
        return cls(pb)

    def __init__(self, pb):
        """Keep this method private."""
        self._sentences = [AnnotatedSentence.from_pb(sent_pb) for sent_pb in pb.sentence]
        for sent in self._sentences:
            sent.document = self

        self._mentions = self.__construct_mentions(pb)

    def __construct_mentions(self, pb):
        mentions = []

        # Get from NER sequence because they tend to be nicer for name
        # mentions. And people only care about name mentions.
        for sentence in self:
            for mention in AnnotatedEntity.from_ner(sentence):
                mentions.append(mention)

        # Get from coref chain
        for chain in pb.corefChain:
            chain_mentions = []
            for mention_pb in chain.mention:
                # If this mention refers to a mention that already
                # exists, use the NER mention instead.
                try:
                    entity = next(mention for mention in mentions if mention.sentence.sentenceIndex == mention_pb.sentenceIndex and mention.head_token == mention_pb.headIndex )
                except StopIteration:
                    entity = AnnotatedEntity(
                        self.sentences[mention_pb.sentenceIndex],
                        (mention_pb.beginIndex, mention_pb.endIndex),
                        mention_pb.headIndex
                        )
                    mentions.append(entity)
                chain_mentions.append(entity)

            # representative mention
            rep_mention = chain_mentions[chain.representative]
            for mention in chain_mentions:
                if mention != rep_mention:
                    mention._canonical_entity = rep_mention
        return mentions

    def __getitem__(self, i):
        return self._sentences[i]

    def __len__(self):
        return len(self._sentences)

    def __str__(self):
        return self.pb.text

    def __repr__(self):
        PREVIEW_LEN = 50
        return "[Document: {}]".format(self.pb.text[:PREVIEW_LEN] + ("..." if len(self.pb.text) > PREVIEW_LEN else ""))

    @ProtobufBacked.json.setter
    def json(self, json_dict):
        self._json = json_dict
        # propagate JSON to children
        for sent, sent_json in izip(self._sentences, json_dict['sentences']):
            sent.json = sent_json

    @classmethod
    def json_to_pb(cls, json_dict):
        sentences = []
        token_idx = 0
        for sent_d in json_dict['sentences']:
            sent_d['tokenOffsetBegin'] = token_idx
            token_idx += len(sent_d['tokens'])
            sent_d['tokenOffsetEnd'] = token_idx
            sent = AnnotatedSentence.json_to_pb(sent_d)
            sentences.append(sent)
        doc = CoreNLP_pb2.Document()
        doc.sentence.extend(sentences)
        return doc

    @staticmethod
    def _reconstruct_text_from_sentence_pbs(sentence_pbs):
        before = lambda sentence_pb: sentence_pb.token[0].before
        after = lambda sentence_pb: sentence_pb.token[-1].after

        text = []
        for i, sent in enumerate(sentence_pbs):
            if i == 0:
                text.append(before(sent))
            text.append(sent.text)
            text.append(after(sent))
        return ''.join(text)

    @property
    def doc_id(self):
        return self.pb.docID

    @property
    def text(self):
        if len(self.pb.text) != 0:
            return self.pb.text

        before = lambda sent: sent[0].before
        after = lambda sent: sent[len(sent) - 1].after

        text = []
        for i, sent in enumerate(self):
            if i == 0:
                text.append(before(sent))
            text.append(sent.text)
            text.append(after(sent))
        return ''.join(text)

    def __getattr__(self, attr):
        """
        If you are looking for an entry in the protobuf that hasn't been
        defined above, this will access it.
        """
        if attr == "_pb":
            raise AttributeError("_pb" is not set)
        return getattr(self.pb, attr)

    @property
    def character_span(self):
        """
        Returns the character span of the sentence
        """
        return (self._sentences[0].character_span[0], self._sentences[-1].character_span[1])

    @property
    def sentences(self):
        return self._sentences

    @property
    def mentions(self):
        """
        Returns all coreferent mentions (as lists of entities)
        """
        return self._mentions

    # These are features that are yet to be supported. In the mean time,
    # users can struggle with the protobuf

# TODO(kelvin): finish specifying the Simple interface for AnnotatedSentence
# http://stanfordnlp.github.io/CoreNLP/simple.html
# In particular, all the methods that take arguments.

# TODO(kelvin): protocol buffers insert undesirable default values. Deal with these somehow.

class AnnotatedSentence(Sentence, ProtobufBacked):
    @classmethod
    def _from_pb(cls, pb):
        # Fill in the text attribute if needed.
        return cls(pb)

    def __init__(self, pb):
        """Keep this method private."""
        self._tokens = [AnnotatedToken.from_pb(tok_pb) for tok_pb in pb.token]

    @property
    def document(self):
        try:
            return self._document
        except AttributeError:
            raise AttributeError("Document has not been set.")

    @document.setter
    def document(self, val):
        self._document = val

    @classmethod
    def _reconstruct_text_from_token_pbs(cls, token_pbs):
        text = []
        for i, tok in enumerate(token_pbs):
            if i != 0:
                text.append(tok.before)
            text.append(tok.word)
        return ''.join(text)

    @ProtobufBacked.json.setter
    def json(self, json_dict):
        self._json = json_dict
        # propagate JSON to children
        for tok, tok_json in izip(self._tokens, json_dict['tokens']):
            tok.json = tok_json

    def __getitem__(self, i):
        return self._tokens[i]

    def __len__(self):
        return len(self._tokens)

    def __str__(self):
        return self.text.encode('utf-8')

    def __unicode__(self):
        return self.text

    def __repr__(self):
        PREVIEW_LEN = 50
        return "[Sentence: {}]".format(self.text[:PREVIEW_LEN] + ("..." if len(self.pb.text) > PREVIEW_LEN else ""))

    @classmethod
    def json_to_pb(cls, json_dict):
        sent = CoreNLP_pb2.Sentence()
        sent.tokenOffsetBegin = json_dict.get('tokenOffsetBegin', 0)
        sent.tokenOffsetEnd = json_dict.get('tokenOffsetEnd', len(json_dict['tokens']))
        tokens = [AnnotatedToken.json_to_pb(d) for d in json_dict['tokens']]
        sent.token.extend(tokens)
        return sent

    @property
    def paragraph(self):
        """
        Returns the paragraph index.
        """
        return self.pb.paragraph

    @property
    def sentenceIndex(self):
        """
        Returns the paragraph index.
        """
        return self.pb.sentenceIndex

    def next_sentence(self):
        """
        Returns the next sentence
        """
        return self.document[self.sentenceIndex + 1]

    def previous_sentence(self):
        """
        Returns the previous sentence
        """
        return self.document[self.sentenceIndex - 1]

    def word(self, i):
        return self._tokens[i].word

    @property
    def before(self):
        return self._tokens[0].before

    @property
    def after(self):
        return self._tokens[-1].after

    @property
    def words(self):
        return [tok.word for tok in self._tokens]

    @property
    def text(self):
        if len(self.pb.text) != 0:
            return self.pb.text

        text = []
        for i, tok in enumerate(self):
            if i != 0:
                text.append(tok.before)
            text.append(tok.word)
        return ''.join(text)

    def pos_tag(self, i):
        return self._tokens[i].pos

    @property
    def pos_tags(self):
        return [tok.pos for tok in self._tokens]

    def lemma(self, i):
        return self._tokens[i].lemma

    @property
    def lemmas(self):
        return [tok.lemma for tok in self._tokens]

    def ner_tag(self, i):
        return self._tokens[i].ner

    @property
    def ner_tags(self):
        return [tok.ner for tok in self._tokens]

    @property
    def tokens(self):
        return self._tokens

    def token(self, i):
        return self._tokens[i]

    def depparse(self, mode="enhancedPlusPlus"):
        """
        Retrieves the appropriate dependency parse.
        Must be one of:
            - basic
            - alternative
            - collapsedCCProcessed
            - collapsed
            - enhanced
            - enhancedPlusPlus
        """
        assert mode in [
            "basic",
            "alternative",
            "collapsedCCProcessed",
            "collapsed",
            "enhanced",
            "enhancedPlusPlus", ], "Invalid mode"
        dep_pb = getattr(self.pb, mode + "Dependencies")
        if dep_pb is None:
            raise AttributeError("No dependencies for mode: " + mode)
        else:
            tree = AnnotatedDependencyParseTree(dep_pb)
            tree.sentence = self
            return tree

    @property
    def character_span(self):
        """
        Returns the character span of the sentence
        """
        return (self._tokens[0].character_span[0], self._tokens[-1].character_span[1])

    def __getattr__(self, attr):
        if attr == "_pb":
            raise AttributeError("_pb" is not set)
        return getattr(self.pb, attr)

    # @property
    # def parse(self):
    #    raise NotImplementedError()

    # @property
    # def natlog_polarities(self):
    #    raise NotImplementedError

    # @property
    # def relations(self, mode="kbp"):
    #    """
    #    Returns any relations found by the annotators.
    #    Valid modes are:
    #        - kbp
    #        - openie
    #        - relation (?)
    #    """
    #    raise NotImplementedError()

    # @property
    # def openie(self):
    #    raise NotImplementedError

    # @property
    # def openie_triples(self):
    #    raise NotImplementedError

    # @property
    # def mentions(self):
    #    """
    #    Supposed to return mentions contained in the sentence.
    #    """
    #    raise NotImplementedError


class AnnotatedToken(Token, ProtobufBacked):
    @classmethod
    def _from_pb(cls, pb):
        return cls()

    def __str__(self):
        return self.pb.word

    def __repr__(self):
        return "[Token: {}]".format(self.pb.word)

    @classmethod
    def json_to_pb(cls, json_dict):
        tok = CoreNLP_pb2.Token()

        def assign_if_present(pb_key, dict_key):
            if dict_key in json_dict:
                setattr(tok, pb_key, json_dict[dict_key])

        mapping = {
            'after': 'after',
            'before': 'before',
            'beginChar': 'characterOffsetBegin',
            'endChar': 'characterOffsetEnd',
            'originalText': 'originalText',
            'word': 'word',
            'pos': 'pos',
            'ner': 'ner',
            'lemma': 'lemma',
            'wikipediaEntity': 'entitylink',
        }

        for pb_key, dict_key in mapping.items():
            assign_if_present(pb_key, dict_key)

        return tok

    @property
    def word(self):
        return self.pb.word

    @property
    def pos(self):
        return self.pb.pos

    @property
    def ner(self):
        return self.pb.ner

    @property
    def lemma(self):
        return self.pb.lemma

    @property
    def originalText(self):
        return self.pb.originalText

    @property
    def before(self):
        return self.pb.before

    @property
    def after(self):
        return self.pb.after

    @property
    def normalized_ner(self):
        return self.pb.normalizedNER

    @property
    def wikipedia_entity(self):
        return self.pb.wikipediaEntity

    @property
    def character_span(self):
        """
        Returns the character span of the token
        """
        return (self.pb.beginChar, self.pb.endChar)


class AnnotatedDependencyParseTree(ProtobufBacked):
    """
    Represents a dependency parse tree
    """
    @classmethod
    def _from_pb(cls, pb):
        return cls(pb)

    def __init__(self, pb):
        self._pb = pb
        self._roots = [r-1 for r in pb.root] # Dependency parses are +1 indexed in the pb.
        self.graph, self.inv_graph = AnnotatedDependencyParseTree.parse_graph(pb.edge)

    def json_to_pb(cls, json_dict):
        raise NotImplementedError

    def __str__(self):
        return str(self.graph)

    def to_json(self):
        """
        Represented as a list of edges:
            dependent: index of child
            dep: dependency label
            governer: index of parent
            dependentgloss: gloss of parent
            governergloss: gloss of parent
        """
        edges = []
        for root in self.roots:
            edges.append({
                'governer': 0,
                'dep': "root",
                'dependent': root+1,
                'governergloss': "root",
                'dependentgloss': self.sentence[root].word,
                })

        for gov, dependents in self.graph.items():
            for dependent, dep in dependents:
                edges.append({
                    'governer': gov+1,
                    'dep': dep,
                    'dependent': dependent+1,
                    'governergloss': self.sentence[gov].word,
                    'dependentgloss': self.sentence[dependent].word,
                    })
        return edges

    @property
    def sentence(self):
        return self._sentence

    @sentence.setter
    def sentence(self, val):
        self._sentence = val

    @staticmethod
    def parse_graph(edges):
        graph = defaultdict(list)
        inv_graph = defaultdict(list)
        for edge in edges:
            graph[edge.source-1].append((edge.target-1, edge.dep))
            inv_graph[edge.target-1].append((edge.source-1, edge.dep))

        return graph, inv_graph

    @property
    def roots(self):
        return self._roots

    def parents(self, i):
        return self.inv_graph[i]

    def children(self, i):
        return self.graph[i]

class AnnotatedEntity(Entity):
    """
    A set of entities
    """
    def __str__(self):
        return self._gloss

    def __repr__(self):
        return "[Entity: {}]".format(self._gloss)

    def __init__(self, sentence, token_span, head_token):
        """
        @arg doc: parent document for this coref mention
        @arg pb: CorefMention protobuf
        """
        self._sentence = sentence
        self._token_span = token_span
        self._head_token = head_token

        token_pbs = sentence.pb.token[token_span[0]:token_span[1]]
        self._gloss = AnnotatedEntity._reconstruct_text_from_token_pbs(token_pbs)
        self._canonical_entity = None

    @classmethod
    def from_ner(cls, sentence):
        # Every change in token type, could be a new entity.
        start_idx, current_ner = 0, 'O'
        for idx, token in enumerate(sentence):
            if token.ner != current_ner:
                if current_ner != 'O':
                    end_idx = idx
                    head_idx = end_idx-1
                    yield AnnotatedEntity(sentence, (start_idx, end_idx), head_idx)
                current_ner = token.ner
                start_idx = idx
        if current_ner != 'O':
            end_idx = len(sentence)
            head_idx = end_idx-1
            yield AnnotatedEntity(sentence, (start_idx, end_idx), head_idx)

    @classmethod
    def _reconstruct_text_from_token_pbs(cls, token_pbs):
        text = []
        for i, tok in enumerate(token_pbs):
            if i != 0:
                text.append(tok.before)
            text.append(tok.word)
        return ''.join(text)

    @property
    def sentence(self):
        """Returns the referring sentence"""
        return self._sentence

    @property
    def token_span(self):
        """Returns the index of the end token."""
        return self._token_span

    @property
    def head_token(self):
        """Returns the index of the end token."""
        return self._head_token

    @property
    def character_span(self):
        """
        Returns the character span of the token
        """
        begin, end = self.token_span
        return (self.sentence[begin].character_span[0], self.sentence[end-1].character_span[-1])

    @property
    def type(self):
        """Returns the type of the string"""
        return self.sentence[self.head_token].ner

    @property
    def gloss(self):
        """Returns the exact string of the entity"""
        return self._gloss

    @property
    def canonical_entity(self):
        """Returns the exact string of the canonical reference"""
        if self._canonical_entity:
            return self._canonical_entity
        else:
            return self

# TODO(kelvin): sentence and doc classes that lazily perform annotations
class LazyDocument(Sentence):
    pass


class LazySentence(Sentence):
    pass
