"""
Client for accessing Stanford CoreNLP in Python
"""

import atexit
import contextlib
import io
import os
import re
import requests
import logging
import json
import shlex
import socket
import subprocess
import time
import sys
import uuid

from datetime import datetime
from pathlib import Path

from six.moves.urllib.parse import urlparse

from stanza.protobuf import Document, parseFromDelimitedString, writeToDelimitedString, to_text
__author__ = 'arunchaganty, kelvinguu, vzhong, wmonroe4'

logger = logging.getLogger('stanza')

# pattern tmp props file should follow
SERVER_PROPS_TMP_FILE_PATTERN = re.compile('corenlp_server-(.*).props')

# Check if str is CoreNLP supported language
CORENLP_LANGS = ['ar', 'arabic', 'chinese', 'zh', 'english', 'en', 'french', 'fr', 'de', 'german', 'es', 'spanish']

LANGUAGE_SHORTHANDS_TO_FULL = {
    "ar": "arabic",
    "zh": "chinese",
    "en": "english",
    "fr": "french",
    "de": "german",
    "es": "spanish"
}


def is_corenlp_lang(props_str):
    """ Check if a string referencs a CoreNLP languauge """
    return props_str.lower() in CORENLP_LANGS


# Validate CoreNLP properties
CORENLP_OUTPUT_VALS = ["conll", "conllu", "json", "serialized", "text", "xml"]


def validate_corenlp_props(properties=None, annotators=None, output_format=None):
    """ Do basic checks to validate CoreNLP properties """
    if output_format and output_format.lower() not in CORENLP_OUTPUT_VALS:
        raise ValueError(f"{output_format} not a valid CoreNLP outputFormat value! Choose from: {CORENLP_OUTPUT_VALS}")
    if type(properties) == dict:
        if "outputFormat" in properties and properties["outputFormat"].lower() not in CORENLP_OUTPUT_VALS:
            raise ValueError(f"{properties['outputFormat']} not a valid CoreNLP outputFormat value! Choose from: "
                             f"{CORENLP_OUTPUT_VALS}")


class AnnotationException(Exception):
    """ Exception raised when there was an error communicating with the CoreNLP server. """
    pass


class TimeoutException(AnnotationException):
    """ Exception raised when the CoreNLP server timed out. """
    pass


class ShouldRetryException(Exception):
    """ Exception raised if the service should retry the request. """
    pass


class PermanentlyFailedException(Exception):
    """ Exception raised if the service should NOT retry the request. """
    pass


def clean_props_file(props_file):
    # check if there is a temp server props file to remove and remove it
    if props_file:
        if os.path.isfile(props_file) and SERVER_PROPS_TMP_FILE_PATTERN.match(os.path.basename(props_file)):
            os.remove(props_file)


class RobustService(object):
    """ Service that resuscitates itself if it is not available. """
    CHECK_ALIVE_TIMEOUT = 120

    def __init__(self, start_cmd, stop_cmd, endpoint, stdout=sys.stdout,
                 stderr=sys.stderr, be_quiet=False, host=None, port=None):
        self.start_cmd = start_cmd and shlex.split(start_cmd)
        self.stop_cmd = stop_cmd and shlex.split(stop_cmd)
        self.endpoint = endpoint
        self.stdout = stdout
        self.stderr = stderr

        self.server = None
        self.is_active = False
        self.be_quiet = be_quiet
        self.host = host
        self.port = port
        atexit.register(self.atexit_kill)

    def is_alive(self):
        try:
            if self.server is not None and self.server.poll() is not None:
                return False
            return requests.get(self.endpoint + "/ping").ok
        except requests.exceptions.ConnectionError as e:
            raise ShouldRetryException(e)

    def start(self):
        if self.start_cmd:
            if self.host and self.port:
                with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                    try:
                        sock.bind((self.host, self.port))
                    except socket.error:
                        raise PermanentlyFailedException("Error: unable to start the CoreNLP server on port %d "
                                                         "(possibly something is already running there)" % self.port)
            if self.be_quiet:
                # Issue #26: subprocess.DEVNULL isn't supported in python 2.7.
                stderr = open(os.devnull, 'w')
            else:
                stderr = self.stderr
            logger.info(f"Starting server with command: {' '.join(self.start_cmd)}")
            self.server = subprocess.Popen(self.start_cmd,
                                           stderr=stderr,
                                           stdout=stderr)

    def atexit_kill(self):
        # make some kind of effort to stop the service (such as a
        # CoreNLP server) at the end of the program.  not waiting so
        # that the python script exiting isn't delayed
        if self.server and self.server.poll() is None:
            self.server.terminate()

    def stop(self):
        if self.server:
            self.server.terminate()
            try:
                self.server.wait(5)
            except subprocess.TimeoutExpired:
                # Resorting to more aggressive measures...
                self.server.kill()
                try:
                    self.server.wait(5)
                except subprocess.TimeoutExpired:
                    # oh well
                    pass
            self.server = None
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
                if self.is_alive():
                    return
                else:
                    self.stop()
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

            if time.time() - start_time < self.CHECK_ALIVE_TIMEOUT:
                time.sleep(1)
            else:
                raise PermanentlyFailedException("Timed out waiting for service to come alive.")

        # At this point we are guaranteed that the service is alive.
        self.is_active = True


def resolve_classpath(classpath):
    """
    Returns the classpath to use for corenlp.

    Prefers to use the given classpath parameter, if available.  If
    not, uses the CORENLP_HOME environment variable.  Resolves $CLASSPATH
    (the exact string) in either the classpath parameter or $CORENLP_HOME.
    """
    if classpath == '$CLASSPATH' or (classpath is None and os.getenv("CORENLP_HOME", None) == '$CLASSPATH'):
        classpath = os.getenv("CLASSPATH")
    elif classpath is None:
        classpath = os.getenv("CORENLP_HOME", os.path.join(str(Path.home()), 'stanza_corenlp'))

        assert os.path.exists(classpath), \
            "Please install CoreNLP by running `stanza.install_corenlp()`. If you have installed it, please define $CORENLP_HOME to be location of your CoreNLP distribution or pass in a classpath parameter."
        classpath = os.path.join(classpath, "*")
    return classpath


class CoreNLPClient(RobustService):
    """ A client to the Stanford CoreNLP server. """

    DEFAULT_ENDPOINT = "http://localhost:9000"
    DEFAULT_TIMEOUT = 60000
    DEFAULT_THREADS = 5
    DEFAULT_OUTPUT_FORMAT = "serialized"
    DEFAULT_MEMORY = "5G"
    DEFAULT_MAX_CHAR_LENGTH = 100000

    def __init__(self, start_server=True,
                 endpoint=DEFAULT_ENDPOINT,
                 timeout=DEFAULT_TIMEOUT,
                 threads=DEFAULT_THREADS,
                 annotators=None,
                 output_format=None,
                 properties=None,
                 stdout=sys.stdout,
                 stderr=sys.stderr,
                 memory=DEFAULT_MEMORY,
                 be_quiet=True,
                 max_char_length=DEFAULT_MAX_CHAR_LENGTH,
                 preload=True,
                 classpath=None,
                 **kwargs):

        # whether or not server should be started by client
        self.start_server = start_server
        self.server_props_path = None
        self.server_start_time = None
        # validate properties
        validate_corenlp_props(properties=properties, annotators=annotators, output_format=output_format)
        # set up client defaults
        self.properties = properties
        self.annotators = annotators
        self.output_format = output_format
        self._setup_client_defaults()
        # start the server
        if start_server:
            # record info for server start
            self.server_start_time = datetime.now()
            self._setup_server_defaults()
            host, port = urlparse(endpoint).netloc.split(":")
            port = int(port)
            assert host == "localhost", "If starting a server, endpoint must be localhost"
            classpath = resolve_classpath(classpath)
            start_cmd = f"java -Xmx{memory} -cp '{classpath}'  edu.stanford.nlp.pipeline.StanfordCoreNLPServer " \
                        f"-port {port} -timeout {timeout} -threads {threads} -maxCharLength {max_char_length} " \
                        f"-quiet {be_quiet} "

            self.server_classpath = classpath
            self.server_host = host
            self.server_port = port

            # set up server defaults
            if self.server_props_path is not None:
                start_cmd += f" -serverProperties {self.server_props_path}"

            # set annotators for server default
            if self.annotators is not None:
                annotators_str = self.annotators if type(annotators) == str else ",".join(annotators)
                start_cmd += f" -annotators {annotators_str}"

            # specify what to preload, if anything
            if preload:
                if type(preload) == bool:
                    # -preload flag means to preload all default annotators
                    start_cmd += " -preload"
                elif type(preload) == list:
                    # turn list into comma separated list string, only preload these annotators
                    start_cmd += f" -preload {','.join(preload)}"
                elif type(preload) == str:
                    # comma separated list of annotators
                    start_cmd += f" -preload {preload}"

            # set outputFormat for server default
            # if no output format requested by user, set to serialized
            start_cmd += f" -outputFormat {self.output_format}"

            # additional options for server:
            # - server_id
            # - ssl
            # - status_port
            # - uriContext
            # - strict
            # - key
            # - username
            # - password
            # - blockList
            for kw in ['ssl', 'strict']:
                if kwargs.get(kw) is not None:
                    start_cmd += f" -{kw}"
            for kw in ['status_port', 'uriContext', 'key', 'username', 'password', 'blockList', 'server_id']:
                if kwargs.get(kw) is not None:
                    start_cmd += f" -{kw} {kwargs.get(kw)}"
            stop_cmd = None
        else:
            start_cmd = stop_cmd = None
            host = port = None

        super(CoreNLPClient, self).__init__(start_cmd, stop_cmd, endpoint,
                                            stdout, stderr, be_quiet, host=host, port=port)

        self.timeout = timeout

    def _setup_client_defaults(self):
        """
        Do some processing of annotators and output_format specified for the client.
        If interacting with an externally started server, these will be defaults for annotate() calls.
        :return: None
        """
        # normalize annotators to str
        if self.annotators is not None:
            self.annotators = self.annotators if type(self.annotators) == str else ",".join(self.annotators)

        # handle case where no output format is specified
        if self.output_format is None:
            if type(self.properties) == dict and 'outputFormat' in self.properties:
                self.output_format = self.properties['outputFormat']
            else:
                self.output_format = CoreNLPClient.DEFAULT_OUTPUT_FORMAT

    def _setup_server_defaults(self):
        """
        Set up the default properties for the server.

        The properties argument can take on one of 3 value types

        1. File path on system or in CLASSPATH (e.g. /path/to/server.props or StanfordCoreNLP-french.properties
        2. Name of a Stanford CoreNLP supported language (e.g. french or fr)
        3. Python dictionary (properties written to tmp file for Java server, erased at end)

        In addition, an annotators list and output_format can be specified directly with arguments. These
        will overwrite any settings in the specified properties.

        If no properties are specified, the standard Stanford CoreNLP English server will be launched. The outputFormat
        will be set to 'serialized' and use the ProtobufAnnotationSerializer.
        """

        # ensure properties is str or dict
        if self.properties is None or (not isinstance(self.properties, str) and not isinstance(self.properties, dict)):
            if self.properties is not None:
                logger.warning('properties passed invalid value (not a str or dict), setting properties = {}')
            self.properties = {}
        # check if properties is a string, pass on to server which can handle
        if isinstance(self.properties, str):
            # try to translate to Stanford CoreNLP language name, or assume properties is a path
            if is_corenlp_lang(self.properties):
                if self.properties.lower() in LANGUAGE_SHORTHANDS_TO_FULL:
                    self.properties = LANGUAGE_SHORTHANDS_TO_FULL[self.properties]
                logger.info(
                    f"Using Stanford CoreNLP default properties for: {self.properties}.  Make sure to have "
                    f"{self.properties} models jar (available for download here: "
                    f"https://stanfordnlp.github.io/CoreNLP/) in CLASSPATH")
            else:
                if not os.path.isfile(self.properties):
                    logger.warning(f"{self.properties} does not correspond to a file path. Make sure this file is in "
                                   f"your CLASSPATH.")
            self.server_props_path = self.properties
        elif isinstance(self.properties, dict):
            # make a copy
            server_start_properties = dict(self.properties)
            if self.annotators is not None:
                server_start_properties['annotators'] = self.annotators
            if self.output_format is not None and isinstance(self.output_format, str):
                server_start_properties['outputFormat'] = self.output_format
            # write desired server start properties to tmp file
            # set up to erase on exit
            tmp_path = write_corenlp_props(server_start_properties)
            logger.info(f"Writing properties to tmp file: {tmp_path}")
            atexit.register(clean_props_file, tmp_path)
            self.server_props_path = tmp_path

    def _request(self, buf, properties, reset_default=False, **kwargs):
        """
        Send a request to the CoreNLP server.

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
            # handle auth
            if 'username' in kwargs and 'password' in kwargs:
                kwargs['auth'] = requests.auth.HTTPBasicAuth(kwargs['username'], kwargs['password'])
                kwargs.pop('username')
                kwargs.pop('password')
            r = requests.post(self.endpoint,
                              params={'properties': str(properties), 'resetDefault': str(reset_default).lower()},
                              data=buf, headers={'content-type': ctype},
                              timeout=(self.timeout*2)/1000, **kwargs)
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            if r.text == "CoreNLP request timed out. Your document may be too long.":
                raise TimeoutException(r.text)
            else:
                raise AnnotationException(r.text)

    def annotate(self, text, annotators=None, output_format=None, properties=None, reset_default=False, **kwargs):
        """
        Send a request to the CoreNLP server.

        :param (str | unicode) text: raw text for the CoreNLPServer to parse
        :param (list | string) annotators: list of annotators to use
        :param (str) output_format: output type from server: serialized, json, text, conll, conllu, or xml
        :param (dict) properties: additional request properties (written on top of defaults)
        :param (bool) reset_default: don't use server defaults

        Precedence for settings:

        1. annotators and output_format args
        2. Values from properties dict
        3. Client defaults self.annotators and self.output_format (set during client construction)
        4. Server defaults

        Additional request parameters (apart from CoreNLP pipeline properties) such as 'username' and 'password'
        can be specified with the kwargs.

        :return: request result
        """

        # validate request properties
        validate_corenlp_props(properties=properties, annotators=annotators, output_format=output_format)
        # set request properties
        request_properties = {}

        # start with client defaults
        if self.annotators is not None:
            request_properties['annotators'] = self.annotators
        if self.output_format is not None:
            request_properties['outputFormat'] = self.output_format

        # add values from properties arg
        # handle str case
        if type(properties) == str:
            if is_corenlp_lang(properties):
                properties = {'pipelineLanguage': properties.lower()}
            else:
                raise ValueError(f"Unrecognized properties keyword {properties}")

        if properties and type(properties) == dict:
            request_properties.update(properties)

        # if annotators list is specified, override with that
        # also can use the annotators field the object was created with
        if annotators is not None and (type(annotators) == str or type(annotators) == list):
            request_properties['annotators'] = annotators if type(self.annotators) == str else ",".join(annotators)

        # if output format is specified, override with that
        if output_format is not None and type(output_format) == str:
            request_properties['outputFormat'] = output_format

        # make the request
        r = self._request(text.encode('utf-8'), request_properties, reset_default, **kwargs)
        if request_properties["outputFormat"] == "json":
            return r.json()
        elif request_properties["outputFormat"] == "serialized":
            doc = Document()
            parseFromDelimitedString(doc, r.content)
            return doc
        elif request_properties["outputFormat"] in ["text", "conllu", "conll", "xml"]:
            return r.text
        else:
            return r

    def update(self, doc, annotators=None, properties=None):
        if properties is None:
            properties = {}
            properties.update({
                'inputFormat': 'serialized',
                'outputFormat': 'serialized',
                'serializer': 'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer'
            })
        if annotators:
            properties['annotators'] = annotators if type(annotators) == str else ",".join(annotators)
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

    def tregex(self, text, pattern, filter=False, annotators=None, properties=None):
        return self.__regex('/tregex', text, pattern, filter, annotators, properties)

    def __regex(self, path, text, pattern, filter, annotators=None, properties=None):
        """
        Send a regex-related request to the CoreNLP server.
        :param (str | unicode) path: the path for the regex endpoint
        :param text: raw text for the CoreNLPServer to apply the regex
        :param (str | unicode) pattern: regex pattern
        :param (bool) filter: option to filter sentences that contain matches, if false returns matches
        :param properties: option to filter sentences that contain matches, if false returns matches
        :return: request result
        """
        self.ensure_alive()
        if properties is None:
            properties = {}
            properties.update({
                'inputFormat': 'text',
                'serializer': 'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer'
            })
        if annotators:
            properties['annotators'] = ",".join(annotators) if isinstance(annotators, list) else annotators

        # force output for regex requests to be json
        properties['outputFormat'] = 'json'

        # TODO: get rid of this once corenlp 4.0.0 is released?
        # the "stupid reason" has hopefully been fixed on the corenlp side
        # but maybe people are married to corenlp 3.9.2 for some reason
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
                },
                data=text.encode('utf-8'),
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
    """
    Transforms tokensregex and semgrex matches to indexed words.
    :param matches: unprocessed regex matches
    :return: flat array of indexed words
    """
    words = [dict(v, **dict([('sentence', i)]))
             for i, s in enumerate(matches['sentences'])
             for k, v in s.items() if k != 'length']
    return words


__all__ = ["CoreNLPClient", "AnnotationException", "TimeoutException", "to_text"]
