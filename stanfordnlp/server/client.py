r"""
Python CoreNLP: a server based interface to Java CoreNLP.
"""
import io
import os
import logging
import json
import shlex
import subprocess
import time
import sys

from six.moves.urllib.parse import urlparse

import requests

from stanfordnlp.protobuf import Document, parseFromDelimitedString, writeToDelimitedString, to_text
__author__ = 'arunchaganty, kelvinguu, vzhong, wmonroe4'

logger = logging.getLogger(__name__)

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

class ShouldRetryException(Exception):
    """
    Exception raised if the service should retry the request.
    """
    pass

class PermanentlyFailedException(Exception):
    """
    Exception raised if the service should retry the request.
    """
    pass

class RobustService(object):
    """
    Service that resuscitates itself if it is not available.
    """
    TIMEOUT = 15

    def __init__(self, start_cmd, stop_cmd, endpoint, stdout=sys.stdout,
                 stderr=sys.stderr, be_quiet=False):
        self.start_cmd = start_cmd and shlex.split(start_cmd)
        self.stop_cmd = stop_cmd and shlex.split(stop_cmd)
        self.endpoint = endpoint
        self.stdout = stdout
        self.stderr = stderr

        self.server = None
        self.is_active = False
        self.be_quiet = be_quiet

    def is_alive(self):
        try:
            return requests.get(self.endpoint + "/ping").ok
        except requests.exceptions.ConnectionError as e:
            raise ShouldRetryException(e)

    def start(self):
        if self.start_cmd:
            if self.be_quiet:
                # Issue #26: subprocess.DEVNULL isn't supported in python 2.7.
                stderr = open(os.devnull, 'w')
            else:
                stderr = self.stderr
            self.server = subprocess.Popen(self.start_cmd,
                                           stderr=stderr,
                                           stdout=stderr)

    def stop(self):
        if self.server:
            self.server.kill()
        if self.stop_cmd:
            subprocess.run(self.stop_cmd, check=True)
        self.is_active = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, _, __, ___):
        self.stop()

    def ensure_alive(self):
        # Check if the service is active and alive
        if self.is_active:
            try:
                return self.is_alive()
            except ShouldRetryException:
                pass

        # If not, try to start up the service.
        if self.server is None:
            self.start()

        # Wait for the service to start up.
        start_time = time.time()
        while True:
            try:
                if self.is_alive():
                    break
            except ShouldRetryException:
                pass

            if time.time() - start_time < self.TIMEOUT:
                time.sleep(1)
            else:
                raise PermanentlyFailedException("Timed out waiting for service to come alive.")

        # At this point we are guaranteed that the service is alive.
        self.is_active = True

class CoreNLPClient(RobustService):
    """
    A CoreNLP client to the Stanford CoreNLP server.
    """
    DEFAULT_ANNOTATORS = "tokenize ssplit lemma pos ner depparse".split()
    DEFAULT_PROPERTIES = {}
    DEFAULT_OUTPUT_FORMAT = "serialized"

    def __init__(self, start_server=True,
                 endpoint="http://localhost:9000",
                 timeout=30000,
                 threads=5,
                 annotators=None,
                 properties=None,
                 output_format=None,
                 stdout=sys.stdout,
                 stderr=sys.stderr,
                 memory="4G",
                 be_quiet=True,
                 max_char_length=100000
                ):
        if isinstance(annotators, str):
            annotators = annotators.split()

        if start_server:
            host, port = urlparse(endpoint).netloc.split(":")
            assert host == "localhost", "If starting a server, endpoint must be localhost"

            assert os.getenv("CORENLP_HOME") is not None, "Please define $CORENLP_HOME where your CoreNLP Java checkout is"
            start_cmd = "java -Xmx{memory} -cp '{corenlp_home}/*'  edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port {port} -timeout {timeout} -threads {threads} -maxCharLength {max_char_length}".format(
                corenlp_home=os.getenv("CORENLP_HOME"),
                port=port,
                memory=memory,
                timeout=timeout,
                threads=threads,
                max_char_length=max_char_length)
            stop_cmd = None
        else:
            start_cmd = stop_cmd = None

        super(CoreNLPClient, self).__init__(start_cmd, stop_cmd, endpoint,
                                            stdout, stderr, be_quiet)
        self.timeout = timeout
        self.default_annotators = annotators or self.DEFAULT_ANNOTATORS
        self.default_properties = properties or self.DEFAULT_PROPERTIES
        self.default_output_format = output_format or self.DEFAULT_OUTPUT_FORMAT

    def _request(self, buf, properties):
        """Send a request to the CoreNLP server.

        :param (str | unicode) text: raw text for the CoreNLPServer to parse
        :param (dict) properties: properties that the server expects
        :return: request result
        """
        self.ensure_alive()

        try:
            input_format = properties.get("inputFormat", "text")
            if input_format == "text":
                ctype = "text/plain; charset=utf-8"
            elif input_format == "serialized":
                ctype = "application/x-protobuf"
            else:
                raise ValueError("Unrecognized inputFormat " + input_format)

            r = requests.post(self.endpoint,
                              params={'properties': str(properties)},
                              data=buf, headers={'content-type': ctype},
                              timeout=(self.timeout*2)/1000)
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            if r.text == "CoreNLP request timed out. Your document may be too long.":
                raise TimeoutException(r.text)
            else:
                raise AnnotationException(r.text)

    def annotate(self, text, annotators=None, output_format=None, properties=None):
        """Send a request to the CoreNLP server.

        :param (str | unicode) text: raw text for the CoreNLPServer to parse
        :param (list | string) annotators: list of annotators to use
        :param (str) output_format: output type from server: serialized, json, text, conll, conllu, or xml
        :param (dict) properties: properties that the server expects
        :return: request result
        """
        # set properties for server call
        if properties is None:
            properties = self.default_properties
            properties.update({
                'annotators': ','.join(annotators or self.default_annotators),
                'inputFormat': 'text',
                'outputFormat': self.default_output_format,
                'serializer': 'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer'
            })
        elif "annotators" not in properties:
            properties.update({'annotators': ','.join(annotators or self.default_annotators)})
        # if an output_format is specified, use that to override
        if output_format is not None:
            properties["outputFormat"] = output_format
        # make the request
        r = self._request(text.encode('utf-8'), properties)
        # customize what is returned based outputFormat
        if properties["outputFormat"] == "serialized":
            doc = Document()
            parseFromDelimitedString(doc, r.content)
            return doc
        elif properties["outputFormat"] == "json":
            return r.json()
        elif properties["outputFormat"] in ["text", "conllu", "conll", "xml"]:
            return r.text
        else:
            return r

    def update(self, doc, annotators=None, properties=None):
        if properties is None:
            properties = self.default_properties
            properties.update({
                'annotators': ','.join(annotators or self.default_annotators),
                'inputFormat': 'serialized',
                'outputFormat': 'serialized',
                'serializer': 'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer'
            })
        with io.BytesIO() as stream:
            writeToDelimitedString(doc, stream)
            msg = stream.getvalue()

        r = self._request(msg, properties)
        doc = Document()
        parseFromDelimitedString(doc, r.content)
        return doc

    def tokensregex(self, text, pattern, filter=False, to_words=False, annotators=None, properties=None):
        # this is required for some reason
        matches = self.__regex('/tokensregex', text, pattern, filter, annotators, properties)
        if to_words:
            matches = regex_matches_to_indexed_words(matches)
        return matches

    def semgrex(self, text, pattern, filter=False, to_words=False, annotators=None, properties=None):
        matches = self.__regex('/semgrex', text, pattern, filter, annotators, properties)
        if to_words:
            matches = regex_matches_to_indexed_words(matches)
        return matches

    def tregrex(self, text, pattern, filter=False, annotators=None, properties=None):
        return self.__regex('/tregex', text, pattern, filter, annotators, properties)

    def __regex(self, path, text, pattern, filter, annotators=None, properties=None):
        """Send a regex-related request to the CoreNLP server.
        :param (str | unicode) path: the path for the regex endpoint
        :param text: raw text for the CoreNLPServer to apply the regex
        :param (str | unicode) pattern: regex pattern
        :param (bool) filter: option to filter sentences that contain matches, if false returns matches
        :param properties: option to filter sentences that contain matches, if false returns matches
        :return: request result
        """
        self.ensure_alive()
        if properties is None:
            properties = self.default_properties
            properties.update({
                'annotators': ','.join(annotators or self.default_annotators),
                'inputFormat': 'text',
                'outputFormat': self.default_output_format,
                'serializer': 'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer'
            })
        elif "annotators" not in properties:
            properties.update({'annotators': ','.join(annotators or self.default_annotators)})

        # HACK: For some stupid reason, CoreNLPServer will timeout if we
        # need to annotate something from scratch. So, we need to call
        # this to ensure that the _regex call doesn't timeout.
        self.annotate(text, properties=properties)

        try:
            # Error occurs unless put properties in params
            input_format = properties.get("inputFormat", "text")
            if input_format == "text":
                ctype = "text/plain; charset=utf-8"
            elif input_format == "serialized":
                ctype = "application/x-protobuf"
            else:
                raise ValueError("Unrecognized inputFormat " + input_format)
            # change request method from `get` to `post` as required by CoreNLP
            r = requests.post(
                self.endpoint + path, params={
                    'pattern': pattern,
                    'filter': filter,
                    'properties': str(properties)
                }, data=text,
                    headers={'content-type': ctype},
                    timeout=(self.timeout*2)/1000,
                    )
            r.raise_for_status()
            return json.loads(r.text)
        except requests.HTTPError as e:
            if r.text.startswith("Timeout"):
                raise TimeoutException(r.text)
            else:
                raise AnnotationException(r.text)
        except json.JSONDecodeError:
            raise AnnotationException(r.text)

def regex_matches_to_indexed_words(matches):
    """Transforms tokensregex and semgrex matches to indexed words.
    :param matches: unprocessed regex matches
    :return: flat array of indexed words
    """
    words = [dict(v, **dict([('sentence', i)]))
             for i, s in enumerate(matches['sentences'])
             for k, v in s.items() if k != 'length']
    return words

__all__ = ["CoreNLPClient", "AnnotationException", "TimeoutException", "to_text"]
