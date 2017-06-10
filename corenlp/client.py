r"""
Python CoreNLP: a server based interface to Java CoreNLP.
"""
import io
import os
import logging
import shlex
import subprocess
import time

from six.moves.urllib.parse import urlparse

import requests

from corenlp_protobuf import Document, parseFromDelimitedString, writeToDelimitedString, to_text
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

    def __init__(self, start_cmd, stop_cmd, endpoint):
        self.start_cmd = start_cmd and shlex.split(start_cmd)
        self.stop_cmd = stop_cmd and shlex.split(stop_cmd)
        self.endpoint = endpoint

        self.server = None
        self.is_active = False

    def is_alive(self):
        try:
            return requests.get(self.endpoint + "/ping").ok
        except requests.exceptions.ConnectionError as e:
            raise ShouldRetryException(e)

    def start(self):
        if self.start_cmd:
            self.server = subprocess.Popen(self.start_cmd)

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

    def __init__(self, start_server=True, endpoint="http://localhost:9000", timeout=5000, annotators=DEFAULT_ANNOTATORS, properties=DEFAULT_PROPERTIES):
        if start_server:
            host, port = urlparse(endpoint).netloc.split(":")
            assert host == "localhost", "If starting a server, endpoint must be localhost"

            assert os.getenv("JAVANLP_HOME") is not None, "Please define $JAVANLP_HOME where your CoreNLP Java checkout is"
            start_cmd = "{javanlp}/bin/javanlp.sh edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port {port} -timeout {timeout}".format(
                javanlp=os.getenv("JAVANLP_HOME"),
                port=port,
                timeout=timeout)
            stop_cmd = None
        else:
            start_cmd = stop_cmd = None

        super(CoreNLPClient, self).__init__(start_cmd, stop_cmd, endpoint)
        self.default_annotators = annotators
        self.default_properties = properties

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
                              data=buf, headers={'content-type': ctype})
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            if r.text == "CoreNLP request timed out. Your document may be too long.":
                raise TimeoutException(r.text)
            else:
                raise AnnotationException(r.text)

    def annotate(self, text, annotators=None, properties=None):
        """Send a request to the CoreNLP server.

        :param (str | unicode) text: raw text for the CoreNLPServer to parse
        :param (dict) properties: properties that the server expects
        :return: request result
        """
        if properties is None:
            properties = self.default_properties
            properties.update({
                'annotators': ','.join(annotators or self.default_annotators),
                'inputFormat': 'text',
                'outputFormat': 'serialized',
                'serializer': 'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer'
            })
        r = self._request(text.encode('utf-8'), properties)
        doc = Document()
        parseFromDelimitedString(doc, r.content)
        return doc

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

__all__ = ["CoreNLPClient", "AnnotationException", "TimeoutException", "to_text"]
