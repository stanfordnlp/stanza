

import stanza
from stanza.protobuf import DependencyEnhancerRequest, Document, Language
from stanza.server.java_protobuf_requests import send_request, add_sentence, JavaProtobufContext

ENHANCER_JAVA = "edu.stanford.nlp.trees.ud.ProcessUniversalEnhancerRequest"

def build_enhancer_request(doc, language, pronouns_pattern):
    if bool(language) == bool(pronouns_pattern):
        raise ValueError("Should set exactly one of language and pronouns_pattern")

    request = DependencyEnhancerRequest()
    if pronouns_pattern:
        request.setRelativePronouns(pronouns_pattern)
    elif language.lower() in ("en", "english"):
        request.language = Language.UniversalEnglish
    elif language.lower() in ("zh", "zh-hans", "chinese"):
        request.language = Language.UniversalChinese
    else:
        raise ValueError("Sorry, but language " + language + " is not supported yet.  Either set a pronouns pattern or file an issue at https://stanfordnlp.github.io/stanza suggesting a mechanism for converting this language")

    request_doc = request.document
    request_doc.text = doc.text
    num_tokens = 0
    for sent_idx, sentence in enumerate(doc.sentences):
        request_sentence = add_sentence(request_doc.sentence, sentence, num_tokens)
        num_tokens = num_tokens + sum(len(token.words) for token in sentence.tokens)

        graph = request_sentence.basicDependencies
        nodes = []
        word_index = 0
        for token in sentence.tokens:
            for word in token.words:
                # TODO: refactor with the bit in java_protobuf_requests
                word_index = word_index + 1
                node = graph.node.add()
                node.sentenceIndex = sent_idx
                node.index = word_index

                if word.head != 0:
                    edge = graph.edge.add()
                    edge.source = word.head
                    edge.target = word_index
                    edge.dep = word.deprel

    return request

def process_doc(doc, language=None, pronouns_pattern=None):
    request = build_enhancer_request(doc, language, pronouns_pattern)
    return send_request(request, Document, ENHANCER_JAVA, "$CLASSPATH")

class UniversalEnhancer(JavaProtobufContext):
    """
    UniversalEnhancer context window

    This is a context window which keeps a process open.  Should allow
    for multiple requests without launching new java processes each time.
    """
    def __init__(self, language=None, pronouns_pattern=None, classpath=None):
        super(UniversalEnhancer, self).__init__(classpath, Document, ENHANCER_JAVA)
        if bool(language) == bool(pronouns_pattern):
            raise ValueError("Should set exactly one of language and pronouns_pattern")
        self.language = language
        self.pronouns_pattern = pronouns_pattern

    def process(self, doc):
        request = build_enhancer_request(doc, self.language, self.pronouns_pattern)
        return self.process_request(request)

def main():
    nlp = stanza.Pipeline('en',
                          processors='tokenize,pos,lemma,depparse')

    with UniversalEnhancer(language="en", classpath="$CLASSPATH") as enhancer:
        doc = nlp("This is the car that I bought")
        result = enhancer.process(doc)
        print(result.sentence[0].enhancedDependencies)

if __name__ == '__main__':
    main()
