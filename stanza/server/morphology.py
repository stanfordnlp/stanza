"""
Direct pipe connection to the Java CoreNLP Morphology class

Only effective for English.  Must be supplied with PTB scheme xpos, not upos
"""


from stanza.protobuf import MorphologyRequest, MorphologyResponse
from stanza.server.java_protobuf_requests import send_request, JavaProtobufContext


MORPHOLOGY_JAVA = "edu.stanford.nlp.process.ProcessMorphologyRequest"

def send_morphology_request(request):
    return send_request(request, MorphologyResponse, MORPHOLOGY_JAVA)

def build_request(words, xpos_tags):
    """
    Turn a list of words and a list of tags into a request

    tags must be xpos, not upos
    """
    request = MorphologyRequest()
    for word, tag in zip(words, xpos_tags):
        tagged_word = request.words.add()
        tagged_word.word = word
        tagged_word.xpos = tag
    return request


def process_text(words, xpos_tags):
    """
    Get the lemmata for each word/tag pair

    Currently the return is a MorphologyResponse from CoreNLP.proto

    tags must be xpos, not upos
    """
    request = build_request(words, xpos_tags)

    return send_morphology_request(request)



class Morphology(JavaProtobufContext):
    """
    Morphology context window

    This is a context window which keeps a process open.  Should allow
    for multiple requests without launching new java processes each time.

    (much faster than calling process_text over and over)
    """
    def __init__(self, classpath=None):
        super(Morphology, self).__init__(classpath, MorphologyResponse, MORPHOLOGY_JAVA)

    def process(self, words, xpos_tags):
        """
        Get the lemmata for each word/tag pair
        """
        request = build_request(words, xpos_tags)
        return self.process_request(request)


def main():
    # TODO: turn this into a unit test, once a new CoreNLP is released
    words    = ["Jennifer", "has",  "the", "prettiest", "antennae"]
    tags     = ["NNP",      "VBZ",  "DT",  "JJS",       "NNS"]
    expected = ["Jennifer", "have", "the", "pretty",    "antenna"]
    result = process_text(words, tags)
    lemma = [x.lemma for x in result.words]
    print(lemma)
    assert lemma == expected

    with Morphology() as morph:
        result = morph.process(words, tags)
        lemma = [x.lemma for x in result.words]
        assert lemma == expected

if __name__ == '__main__':
    main()
