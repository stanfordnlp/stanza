"""
Tests that call a running CoreNLPClient.
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import multiprocessing
import pytest
import requests
import stanza.server as corenlp
import stanza.server.client as client
import shlex
import subprocess
import time

from stanza.tests import *

# set the marker for this module
pytestmark = [pytest.mark.travis, pytest.mark.client]

TEXT = "Chris wrote a simple sentence that he parsed with Stanford CoreNLP.\n"

MAX_REQUEST_ATTEMPTS = 5

EN_GOLD = """
Sentence #1 (12 tokens):
Chris wrote a simple sentence that he parsed with Stanford CoreNLP.

Tokens:
[Text=Chris CharacterOffsetBegin=0 CharacterOffsetEnd=5 PartOfSpeech=NNP]
[Text=wrote CharacterOffsetBegin=6 CharacterOffsetEnd=11 PartOfSpeech=VBD]
[Text=a CharacterOffsetBegin=12 CharacterOffsetEnd=13 PartOfSpeech=DT]
[Text=simple CharacterOffsetBegin=14 CharacterOffsetEnd=20 PartOfSpeech=JJ]
[Text=sentence CharacterOffsetBegin=21 CharacterOffsetEnd=29 PartOfSpeech=NN]
[Text=that CharacterOffsetBegin=30 CharacterOffsetEnd=34 PartOfSpeech=WDT]
[Text=he CharacterOffsetBegin=35 CharacterOffsetEnd=37 PartOfSpeech=PRP]
[Text=parsed CharacterOffsetBegin=38 CharacterOffsetEnd=44 PartOfSpeech=VBD]
[Text=with CharacterOffsetBegin=45 CharacterOffsetEnd=49 PartOfSpeech=IN]
[Text=Stanford CharacterOffsetBegin=50 CharacterOffsetEnd=58 PartOfSpeech=NNP]
[Text=CoreNLP CharacterOffsetBegin=59 CharacterOffsetEnd=66 PartOfSpeech=NNP]
[Text=. CharacterOffsetBegin=66 CharacterOffsetEnd=67 PartOfSpeech=.]
""".strip()

class HTTPMockServerTimeoutContext:
    """ For launching an HTTP server on certain port with an specified delay at responses """
    def __init__(self, port, timeout_secs):
        self.port = port
        self.timeout_secs = timeout_secs

    def __enter__(self):
        class HTTPTimeoutHandler(BaseHTTPRequestHandler):
            def do_POST(self_inner):
                time.sleep(self.timeout_secs)
                self_inner.send_response(200)
                self_inner.send_header('Content-type', 'text/plain; charset=utf-8')
                self_inner.end_headers()
                self_inner.wfile.write("HTTPMockServerTimeout")
        def run_webserver():
            HTTPServer(('127.0.0.1',self.port), HTTPTimeoutHandler).serve_forever()

        self.p = multiprocessing.Process(target=run_webserver, args=())
        self.p.daemon = True
        self.p.start()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.p.terminate()

class TestCoreNLPClient:
    @pytest.fixture(scope="class")
    def corenlp_client(self):
        """ Client to run tests on """
        client = corenlp.CoreNLPClient(annotators='tokenize,ssplit,pos,lemma,ner,depparse',
                                       server_id='stanza_main_test_server')
        yield client
        client.stop()


    def test_connect(self, corenlp_client):
        corenlp_client.ensure_alive()
        assert corenlp_client.is_active
        assert corenlp_client.is_alive()


    def test_context_manager(self):
        with corenlp.CoreNLPClient(annotators="tokenize,ssplit",
                                   endpoint="http://localhost:9001") as context_client:
            ann = context_client.annotate(TEXT)
            assert corenlp.to_text(ann.sentence[0]) == TEXT[:-1]

    def test_no_duplicate_servers(self):
        """We expect a second server on the same port to fail"""
        with pytest.raises(corenlp.PermanentlyFailedException):
            with corenlp.CoreNLPClient(annotators="tokenize,ssplit") as duplicate_server:
                raise RuntimeError("This should have failed")

    def test_annotate(self, corenlp_client):
        ann = corenlp_client.annotate(TEXT)
        assert corenlp.to_text(ann.sentence[0]) == TEXT[:-1]


    def test_update(self, corenlp_client):
        ann = corenlp_client.annotate(TEXT)
        ann = corenlp_client.update(ann)
        assert corenlp.to_text(ann.sentence[0]) == TEXT[:-1]


    def test_tokensregex(self, corenlp_client):
        pattern = '([ner: PERSON]+) /wrote/ /an?/ []{0,3} /sentence|article/'
        matches = corenlp_client.tokensregex(TEXT, pattern)
        assert len(matches["sentences"]) == 1
        assert matches["sentences"][0]["length"] == 1
        assert matches == {
            "sentences": [{
                "0": {
                    "text": "Chris wrote a simple sentence",
                    "begin": 0,
                    "end": 5,
                    "1": {
                        "text": "Chris",
                        "begin": 0,
                        "end": 1
                    }},
                "length": 1
            },]}


    def test_semgrex(self, corenlp_client):
        pattern = '{word:wrote} >nsubj {}=subject >obj {}=object'
        matches = corenlp_client.semgrex(TEXT, pattern, to_words=True)
        assert matches == [
            {
                "text": "wrote",
                "begin": 1,
                "end": 2,
                "$subject": {
                    "text": "Chris",
                    "begin": 0,
                    "end": 1
                },
                "$object": {
                    "text": "sentence",
                    "begin": 4,
                    "end": 5
                },
                "sentence": 0,}]


    def test_external_server_legacy_start_server(self):
        """ Test starting up an external server and accessing with a client with start_server=False """
        corenlp_home = client.resolve_classpath(None)
        start_cmd = f'java -Xmx5g -cp "{corenlp_home}" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 ' \
                    f'-timeout 60000 -server_id stanza_external_server -serverProperties {SERVER_TEST_PROPS}'
        start_cmd = start_cmd and shlex.split(start_cmd)
        external_server_process = subprocess.Popen(start_cmd)
        with corenlp.CoreNLPClient(start_server=False, endpoint="http://localhost:9001") as external_server_client:
            ann = external_server_client.annotate(TEXT, annotators='tokenize,ssplit,pos', output_format='text')
        assert external_server_process
        external_server_process.terminate()
        external_server_process.wait(5)
        assert ann.strip() == EN_GOLD

    def test_external_server_available(self):
        """ Test starting up an external available server and accessing with a client with start_server=StartServer.DONT_START """
        corenlp_home = os.getenv('CORENLP_HOME')
        start_cmd = f'java -Xmx5g -cp "{corenlp_home}/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 ' \
                    f'-timeout 60000 -server_id stanza_external_server -serverProperties {SERVER_TEST_PROPS}'
        start_cmd = start_cmd and shlex.split(start_cmd)
        external_server_process = subprocess.Popen(start_cmd)
        time.sleep(5) # wait and make sure the external CoreNLP server is up and running
        with corenlp.CoreNLPClient(start_server=corenlp.StartServer.DONT_START, endpoint="http://localhost:9001") as external_server_client:
            ann = external_server_client.annotate(TEXT, annotators='tokenize,ssplit,pos', output_format='text')
        assert external_server_process
        external_server_process.terminate()
        external_server_process.wait(5)
        assert ann.strip() == EN_GOLD

    def test_external_server_unavailable(self):
        """ Test accessing with a client with start_server=StartServer.DONT_START to an external unavailable server """
        with pytest.raises(corenlp.AnnotationException):
            with corenlp.CoreNLPClient(start_server=corenlp.StartServer.DONT_START, endpoint="http://localhost:9001") as external_server_client:
                ann = external_server_client.annotate(TEXT, annotators='tokenize,ssplit,pos', output_format='text')

    def test_external_server_timeout(self):
        """ Test starting up an external server with long response time (20 seconds) and accessing with a client with start_server=StartServer.DONT_START and timeout=5000"""
        with HTTPMockServerTimeoutContext(9001, 20):
            time.sleep(5) # wait and make sure the external HTTPMockServer server is up and running
            with pytest.raises(corenlp.TimeoutException):
                with corenlp.CoreNLPClient(start_server=corenlp.StartServer.DONT_START, endpoint="http://localhost:9001", timeout=5000) as external_server_client:
                    ann = external_server_client.annotate(TEXT, annotators='tokenize,ssplit,pos', output_format='text')

    def test_external_server_try_start_with_external(self):
        """ Test starting up an external server and accessing with a client with start_server=StartServer.TRY_START """
        corenlp_home = os.getenv('CORENLP_HOME')
        start_cmd = f'java -Xmx5g -cp "{corenlp_home}/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 ' \
                    f'-timeout 60000 -server_id stanza_external_server -serverProperties {SERVER_TEST_PROPS}'
        start_cmd = start_cmd and shlex.split(start_cmd)
        external_server_process = subprocess.Popen(start_cmd)
        with corenlp.CoreNLPClient(start_server=corenlp.StartServer.TRY_START,
                                   annotators='tokenize,ssplit,pos',
                                   endpoint="http://localhost:9001") as external_server_client:
            ann = external_server_client.annotate(TEXT, annotators='tokenize,ssplit,pos', output_format='text')
        assert external_server_process
        external_server_process.terminate()
        external_server_process.wait(5)
        assert ann.strip() == EN_GOLD

    def test_external_server_try_start(self):
        """ Test starting up a server with a client with start_server=StartServer.TRY_START """
        corenlp_home = os.getenv('CORENLP_HOME')
        with corenlp.CoreNLPClient(start_server=corenlp.StartServer.TRY_START,
                                   annotators='tokenize,ssplit,pos',
                                   endpoint="http://localhost:9001") as external_server_client:
            ann = external_server_client.annotate(TEXT, annotators='tokenize,ssplit,pos', output_format='text')
        assert ann.strip() == EN_GOLD

    def test_external_server_force_start(self):
        """ Test starting up an external server and accessing with a client with start_server=StartServer.FORCE_START """
        corenlp_home = os.getenv('CORENLP_HOME')
        start_cmd = f'java -Xmx5g -cp "{corenlp_home}/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 ' \
                    f'-timeout 60000 -server_id stanza_external_server -serverProperties {SERVER_TEST_PROPS}'
        start_cmd = start_cmd and shlex.split(start_cmd)
        external_server_process = subprocess.Popen(start_cmd)
        time.sleep(5) # wait and make sure the external CoreNLP server is up and running
        with pytest.raises(corenlp.PermanentlyFailedException):
            with corenlp.CoreNLPClient(start_server=corenlp.StartServer.FORCE_START, endpoint="http://localhost:9001") as external_server_client:
                ann = external_server_client.annotate(TEXT, annotators='tokenize,ssplit,pos', output_format='text')
        assert external_server_process
        external_server_process.terminate()
        external_server_process.wait(5)
