"""
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

# pattern tmp props file should follow
SERVER_PROPS_TMP_FILE_PATTERN = re.compile('corenlp_server-(.*).props')

# info for Stanford CoreNLP supported languages
LANGUAGE_SHORTHANDS_TO_FULL = {
    "ar": "arabic",
    "zh": "chinese",
    "fr": "french",
    "de": "german",
    "es": "spanish"
}

LANGUAGE_DEFAULT_ANNOTATORS = {
    "arabic": "tokenize,ssplit,pos,parse",
    "chinese": "tokenize,ssplit,pos,lemma,ner,parse,coref",
    "french": "tokenize,ssplit,pos,depparse",
    "german": "tokenize,ssplit,pos,ner,parse",
    "spanish": "tokenize,ssplit,pos,lemma,ner,depparse,kbp"
}


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
    """ Exception raised if the service should retry the request. """
    pass


class RobustService(object):
    """ Service that resuscitates itself if it is not available. """
    CHECK_ALIVE_TIMEOUT = 120

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
            print(f"Starting server with command: {' '.join(self.start_cmd)}")
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

            if time.time() - start_time < self.CHECK_ALIVE_TIMEOUT:
                time.sleep(1)
            else:
                raise PermanentlyFailedException("Timed out waiting for service to come alive.")

        # At this point we are guaranteed that the service is alive.
        self.is_active = True


class CoreNLPClient(RobustService):
    """ A CoreNLP client to the Stanford CoreNLP server. """

    DEFAULT_ENDPOINT = "http://localhost:9000"
    DEFAULT_TIMEOUT = 60000
    DEFAULT_THREADS = 5
    DEFAULT_ANNOTATORS = "tokenize,ssplit,pos,lemma,ner,depparse"
    DEFAULT_OUTPUT_FORMAT = "serialized"
    DEFAULT_MEMORY = "5G"
    DEFAULT_MAX_CHAR_LENGTH = 100000
    DEFAULT_SERIALIZER = "edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer"
    DEFAULT_INPUT_FORMAT = "text"
    PIPELINE_LANGUAGES = \
        ['ar', 'arabic', 'chinese', 'zh', 'english', 'en', 'french', 'fr', 'de', 'german', 'es', 'spanish']

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
                 preload=True,
                 **kwargs):

        # properties cache maps keys to properties dictionaries for convenience
        self.properties_cache = {}
        self.server_props_file = {'is_temp': False, 'path': None}
        # start the server
        if start_server:
            # set up default properties for server
            self._setup_default_server_props(properties, annotators, output_format)
            # at this point self.server_start_info and self.server_props_file should be set
            host, port = urlparse(endpoint).netloc.split(":")
            assert host == "localhost", "If starting a server, endpoint must be localhost"
            corenlp_home = os.getenv("CORENLP_HOME")
            assert corenlp_home is not None, \
                "Please define $CORENLP_HOME to be location of your CoreNLP Java checkout"
            start_cmd = f"java -Xmx{memory} -cp '{corenlp_home}/*'  edu.stanford.nlp.pipeline.StanfordCoreNLPServer " \
                        f"-port {port} -timeout {timeout} -threads {threads} -maxCharLength {max_char_length} " \
                        f"-quiet {be_quiet} -serverProperties {self.server_props_file['path']}"
            if preload and self.server_start_info.get('preload_annotators'):
                start_cmd += f" -preload {self.server_start_info['preload_annotators']}"
            # additional options for server:
            # - server_id
            # - ssl
            # - status_port
            # - uriContext
            # - strict
            # - key
            # - username
            # - password
            # - blacklist
            for kw in ['ssl', 'strict']:
                if kwargs.get(kw) is not None:
                    start_cmd += f" -{kw}"
            for kw in ['status_port', 'uriContext', 'key', 'username', 'password', 'blacklist', 'server_id']:
                if kwargs.get(kw) is not None:
                    start_cmd += f" -{kw} {kwargs.get(kw)}"
            stop_cmd = None
        else:
            start_cmd = stop_cmd = None
            self.server_start_info = {}

        super(CoreNLPClient, self).__init__(start_cmd, stop_cmd, endpoint,
                                            stdout, stderr, be_quiet)

        self.timeout = timeout

    def _setup_default_server_props(self, properties, annotators, output_format):
        """
        Set up the default properties for the server from either:

        1. File path on system or in CLASSPATH (e.g. /path/to/server.props or StanfordCoreNLP-french.properties
        2. Stanford CoreNLP supported language (e.g. french)
        3. Python dictionary (properties written to tmp file for Java server, erased at end)
        4. Default (just use standard defaults set server side in Java code, with the exception that the default
                    default outputFormat is changed to serialized)

        If defaults are being set client side, values of annotators and output_format will overwrite the
        client side properties.  If the defaults are being set server side, those parameters will be ignored.

        Info about the properties used to start the server is stored in self.server_start_info
        If a file is used, info about the file (path, whether tmp or not) is stored in self.server_props_file
        """
        # store information about server start up
        self.server_start_info = {}
        # ensure properties is str or dict
        if properties is None or (not isinstance(properties, str) and not isinstance(properties, dict)):
            if properties is not None:
                print('Warning: properties passed invalid value (not a str or dict), setting properties = {}')
            properties = {}
        # check if properties is a string
        if isinstance(properties, str):
            # translate Stanford CoreNLP language name to properties file if properties is a language name
            if properties.lower() in CoreNLPClient.PIPELINE_LANGUAGES:
                lang_name = properties.lower()
                if lang_name in LANGUAGE_SHORTHANDS_TO_FULL:
                    lang_name = LANGUAGE_SHORTHANDS_TO_FULL[lang_name]
                self.server_props_file['path'] = f'StanfordCoreNLP-{lang_name}.properties'
                self.server_start_info['preload_annotators'] = LANGUAGE_DEFAULT_ANNOTATORS[lang_name]
                print(f"Using Stanford CoreNLP default properties for: {lang_name}.  Make sure to have {lang_name} "
                      f"models jar (available for download here: https://stanfordnlp.github.io/CoreNLP/) in CLASSPATH")
            # otherwise assume properties string is a path
            else:
                self.server_props_file['path'] = properties
                if os.path.isfile(properties):
                    props_from_file = read_corenlp_props(properties)
                    self.server_start_info['props'] = props_from_file
                    self.server_start_info['preload_annotators'] = props_from_file.get('annotators')
                else:
                    print(f"Warning: {properties} does not correspond to a file path.")
            print(f"Setting server defaults from: {self.server_props_file['path']}")
            self.server_start_info['props_file'] = self.server_props_file['path']
            self.server_start_info['server_side'] = True
            if annotators is not None:
                print(f"Warning: Server defaults being set server side, ignoring annotators={annotators}")
            if output_format is not None:
                print(f"Warning: Server defaults being set server side, ignoring output_format={output_format}")
        # check if client side should set default properties
        else:
            # set up properties from client side
            # the Java Stanford CoreNLP server defaults to "json" for outputFormat
            # but by default servers started by Python interface will override this to return serialized object
            client_side_properties = {
                'annotators': CoreNLPClient.DEFAULT_ANNOTATORS,
                'outputFormat': CoreNLPClient.DEFAULT_OUTPUT_FORMAT,
                'serializer': CoreNLPClient.DEFAULT_SERIALIZER
            }
            client_side_properties.update(properties)
            # override if a specific annotators list was specified
            if annotators:
                client_side_properties['annotators'] = \
                    ",".join(annotators) if isinstance(annotators, list) else annotators
            # override if a specific output format was specified
            if output_format is not None and isinstance(output_format, str):
                client_side_properties['outputFormat'] = output_format
            # write client side props to a tmp file which will be erased at end
            self.server_props_file['path'] = write_corenlp_props(client_side_properties)
            self.server_props_file['is_temp'] = True
            # record server start up info
            self.server_start_info['client_side'] = True
            self.server_start_info['props'] = client_side_properties
            self.server_start_info['props_file'] = self.server_props_file['path']
            self.server_start_info['preload_annotators'] = client_side_properties['annotators']

    def stop(self):
        # check if there is a temp server props file to remove and remove it
        if self.server_props_file['is_temp']:
            if os.path.isfile(self.server_props_file['path']) and \
                    SERVER_PROPS_TMP_FILE_PATTERN.match(os.path.basename(self.server_props_file['path'])):
                os.remove(self.server_props_file['path'])
        # run base class stop
        super(CoreNLPClient, self).stop()

    def _request(self, buf, properties, **kwargs):
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
                              params={'properties': str(properties)},
                              data=buf, headers={'content-type': ctype},
                              timeout=(self.timeout*2)/1000, **kwargs)
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            if r.text == "CoreNLP request timed out. Your document may be too long.":
                raise TimeoutException(r.text)
            else:
                raise AnnotationException(r.text)

    def register_properties_key(self, key, props):
        """ Register a properties dictionary with a key in the client's properties_cache """
        if key in CoreNLPClient.PIPELINE_LANGUAGES:
            print(f'Key {key} not registered in properties cache.  Names of Stanford CoreNLP supported languages are '
                  f'reserved for Stanford CoreNLP defaults for that language.  For instance "french" or "fr" '
                  f'corresponds to using the defaults in StanfordCoreNLP-french.properties which are stored with the '
                  f'server.  If you want to store custom defaults for that language, it is suggested to use a key like '
                  f' "fr-custom", etc...'
                  )
        else:
            self.properties_cache[key] = props

    def annotate(self, text, annotators=None, output_format=None, properties_key=None, properties=None, **kwargs):
        """
        Send a request to the CoreNLP server.

        :param (str | unicode) text: raw text for the CoreNLPServer to parse
        :param (list | string) annotators: list of annotators to use
        :param (str) output_format: output type from server: serialized, json, text, conll, conllu, or xml
        :param (str) properties_key: key into properties cache for the client
        :param (dict) properties: additional request properties (written on top of defaults)

        The properties for a request are written in this order:

        1. Server default properties (server side)
        2. Properties from client's properties_cache corresponding to properties_key (client side)
           If the properties_key is the name of a Stanford CoreNLP supported language:
           [Arabic, Chinese, English, French, German, Spanish], the Stanford CoreNLP defaults will be used (server side)
        3. Additional properties corresponding to properties (client side)
        4. Special case specific properties: annotators, output_format (client side)

        :return: request result
        """
        # set properties for server call
        # first look for a cached default properties set
        # if a Stanford CoreNLP supported language is specified, just pass {pipelineLanguage="french"}
        if properties_key is not None:
            if properties_key.lower() in CoreNLPClient.PIPELINE_LANGUAGES:
                request_properties = {'pipelineLanguage': properties_key.lower()}
            else:
                request_properties = self.properties_cache.get(properties_key, {})
        else:
            request_properties = {}
        # add on custom properties for this request
        if properties is None:
            properties = {}
        request_properties.update(properties)
        # if annotators list is specified, override with that
        if annotators is not None:
            request_properties['annotators'] = ",".join(annotators) if isinstance(annotators, list) else annotators
        # always send an output format with request
        # in some scenario's the server's default output format is unknown, so default to serialized
        if output_format is not None:
            request_properties['outputFormat'] = output_format
        if request_properties.get('outputFormat') is None:
            if self.server_start_info.get('props', {}).get('outputFormat'):
                request_properties['outputFormat'] = self.server_start_info['props']['outputFormat']
            else:
                request_properties['outputFormat'] = CoreNLPClient.DEFAULT_OUTPUT_FORMAT
        # make the request
        r = self._request(text.encode('utf-8'), request_properties, **kwargs)
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
            properties['annotators'] = ",".join(annotators) if isinstance(annotators, list) else annotators
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
