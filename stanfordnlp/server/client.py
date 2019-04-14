r"""
Client for accessing Stanford CoreNLP in Python
"""
import io
import os
import re
import requests
import logging
import json
import shlex
import subprocess
import time
import sys
import uuid

from six.moves.urllib.parse import urlparse

from stanfordnlp.protobuf import Document, parseFromDelimitedString, writeToDelimitedString, to_text
__author__ = 'arunchaganty, kelvinguu, vzhong, wmonroe4'

logger = logging.getLogger(__name__)

SERVER_PROPS_TMP_FILE_PATTERN = re.compile('corenlp_server-(.*).props')


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

    DEFAULT_ENDPOINT = "http://localhost:9000"
    DEFAULT_TIMEOUT = 60000
    DEFAULT_THREADS = 5
    DEFAULT_ANNOTATORS = "tokenize ssplit pos lemma ner depparse".split()
    DEFAULT_PROPERTIES = {}
    DEFAULT_OUTPUT_FORMAT = "serialized"
    DEFAULT_MEMORY = "4G"
    DEFAULT_MAX_CHAR_LENGTH = 100000
    DEFAULT_SERIALIZER = "edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer"
    DEFAULT_INPUT_FORMAT = "text"

    def __init__(self, start_server=True,
                 endpoint=DEFAULT_ENDPOINT,
                 timeout=DEFAULT_TIMEOUT,
                 threads=DEFAULT_THREADS,
                 annotators=None,
                 properties=None,
                 output_format=None,
                 stdout=sys.stdout,
                 stderr=sys.stderr,
                 memory=DEFAULT_MEMORY,
                 be_quiet=True,
                 max_char_length=DEFAULT_MAX_CHAR_LENGTH,
                 preload=True
                 ):

        # process server options
        # set up default properties
        self.server_props_file = {'is_temp': False, 'file_path': None}
        if properties is None:
            self.default_properties = dict(CoreNLPClient.DEFAULT_PROPERTIES)
        elif isinstance(properties, str):
            if os.path.isfile(properties):
                # assume properties is a file path with Stanford CoreNLP properties
                self.default_properties = read_corenlp_props(properties)
            else:
                print("Warning: {properties} cannot be found on filesystem")
            self.server_props_file = {'is_temp': False, 'file_path': properties}
        elif isinstance(properties, dict):
            self.default_properties = properties
        else:
            raise RuntimeError(f"Error: {properties} is not a valid type for CoreNLPClient properties.")
        # set up default annotators
        if annotators is None:
            if self.default_properties.get('annotators') is None:
                self.default_annotators = CoreNLPClient.DEFAULT_ANNOTATORS
            else:
                self.default_annotators = self.default_properties.get('annotators')
        else:
            self.default_annotators = annotators
        if isinstance(self.default_annotators, str):
            self.default_annotators = self.default_annotators.split()
        self.default_properties['annotators'] = self.default_annotators
        # set up default output format
        if output_format is None:
            if self.default_properties.get('outputFormat') is None:
                self.default_output_format = CoreNLPClient.DEFAULT_OUTPUT_FORMAT
            else:
                self.default_output_format = self.default_properties.get('outputFormat')
        else:
            self.default_output_format = output_format
        self.default_properties['outputFormat'] = self.default_output_format
        # set up default serializer
        if self.default_properties.get('serializer') is None:
            self.default_properties['serializer'] = CoreNLPClient.DEFAULT_SERIALIZER
        # set up default input format
        if self.default_properties.get('inputFormat') is None:
            self.default_properties['inputFormat'] = CoreNLPClient.DEFAULT_INPUT_FORMAT

        # if necessary write server props to tmp file
        if self.server_props_file['file_path'] is None:
            self.server_props_file['file_path'] = write_corenlp_props(self.default_properties)
            self.server_props_file['is_temp'] = True

        if start_server:
            host, port = urlparse(endpoint).netloc.split(":")
            assert host == "localhost", "If starting a server, endpoint must be localhost"
            corenlp_home = os.getenv("CORENLP_HOME")
            assert corenlp_home is not None, \
                "Please define $CORENLP_HOME to be location of your CoreNLP Java checkout"
            start_cmd = f"java -Xmx{memory} -cp '{corenlp_home}/*'  edu.stanford.nlp.pipeline.StanfordCoreNLPServer " \
                        f"-port {port} -timeout {timeout} -threads {threads} -maxCharLength {max_char_length} " \
                        f"-quiet {be_quiet} -serverProperties {self.server_props_file['file_path']}"
            if preload and self.default_annotators:
                start_cmd += f" -preload {','.join(self.default_annotators)}"
            print("starting server with command: " + start_cmd)
            stop_cmd = None
        else:
            start_cmd = stop_cmd = None

        super(CoreNLPClient, self).__init__(start_cmd, stop_cmd, endpoint,
                                            stdout, stderr, be_quiet)

        self.timeout = timeout

        if start_server and preload:
            self.start()

    def stop(self):
        # check if there is a temp server props file to remove and remove
        if self.server_props_file['is_temp']:
            if os.path.isfile(self.server_props_file['file_path']) and \
                    SERVER_PROPS_TMP_FILE_PATTERN.match(os.path.basename(self.server_props_file['file_path'])):
                os.remove(self.server_props_file['file_path'])
        # run base class stop
        super(CoreNLPClient, self).stop()

    def _request(self, buf, properties):
        """Send a request to the CoreNLP server.

        :param (str | bytes) buf: data to be sent with the request
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
            properties = {}
        # if annotators list is specified, override with that
        if annotators is not None:
            if isinstance(annotators, str):
                annotators = annotators.split()
            properties['annotators'] = annotators
        # if an output_format is specified, use that to override
        if output_format is not None:
            properties["outputFormat"] = output_format
        else:
            properties["outputFormat"] = self.default_output_format
        # make the request
        r = self._request(text.encode('utf-8'), properties)
        if properties.get("outputFormat") is None or properties["outputFormat"] == "json":
            return r.json()
        elif properties["outputFormat"] == "serialized":
            doc = Document()
            parseFromDelimitedString(doc, r.content)
            return doc
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
                }, data=text.encode('utf-8'),
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


def read_corenlp_props(props_path):
    """ Read a Stanford CoreNLP properties file into a dict """
    props_dict = {}
    if os.path.exists(props_path):
        with open(props_path) as props_file:
            entry_lines = \
                [entry_line for entry_line in props_file.read().split('\n')
                 if entry_line.strip() and not entry_line.startswith('#')]
            for entry_line in entry_lines:
                k = entry_line.split('=')[0]
                k_len = len(k+"=")
                v = entry_line[k_len:]
                props_dict[k.strip()] = v
        return props_dict

    else:
        raise RuntimeError(f'Error: Properties file at {props_path} does not exist!')


def write_corenlp_props(props_dict, file_path=None):
    """ Write a Stanford CoreNLP properties dict to a file """
    if file_path is None:
        file_path = f"corenlp_server-{uuid.uuid4().hex[:16]}.props"
        # confirm tmp file path matches pattern
        assert SERVER_PROPS_TMP_FILE_PATTERN.match(file_path)
    with open(file_path, 'w') as props_file:
        for k, v in props_dict.items():
            if isinstance(v, list):
                writeable_v = ",".join(v)
            else:
                writeable_v = v
            props_file.write(f'{k} = {writeable_v}\n\n')
    return file_path


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
