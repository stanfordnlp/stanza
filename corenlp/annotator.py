"""
Defines a base class that can be used to annotate.
"""
import io
from multiprocessing import Process
from six.moves.BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from six.moves import http_client as HTTPStatus

from corenlp_protobuf import Document, parseFromDelimitedString, writeToDelimitedString

class Annotator(Process):
    """
    This annotator base class hosts a lightweight server that accepts
    annotation requests from CoreNLP.
    Each annotator simply defines 3 functions: requires, provides and annotate.

    This class takes care of defining appropriate endpoints to interface
    with CoreNLP.
    """
    @property
    def name(self):
        """
        Name of the annotator (used by CoreNLP)
        """
        raise NotImplementedError()

    @property
    def requires(self):
        """
        Requires has to specify all the annotations required before we
        are called.
        """
        raise NotImplementedError()

    @property
    def provides(self):
        """
        The set of annotations guaranteed to be provided when we are done.
        NOTE: that these annotations are either fully qualified Java
        class names or refer to nested classes of
        edu.stanford.nlp.ling.CoreAnnotations (as is the case below).
        """
        raise NotImplementedError()

    def annotate(self, ann):
        """
        @ann: is a protobuf annotation object.
        Actually populate @ann with tokens.
        """
        raise NotImplementedError()

    @property
    def properties(self):
        """
        Defines a Java property to define this anntoator to CoreNLP.
        """
        return {
            "customAnnotatorClass.{}".format(self.name): "edu.stanford.nlp.pipeline.GenericWebServiceAnnotator",
            "generic.endpoint": "http://{}:{}".format(self.host, self.port),
            "generic.requires": ",".join(self.requires),
            "generic.provides": ",".join(self.provides),
            }

    class _Handler(BaseHTTPRequestHandler):
        annotator = None

        def __init__(self, request, client_address, server):
            BaseHTTPRequestHandler.__init__(self, request, client_address, server)

        def do_GET(self):
            """
            Handle a ping request
            """
            if not self.path.endswith("/"): self.path += "/"
            if self.path == "/ping/":
                msg = "pong".encode("UTF-8")

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/application")
                self.send_header("Content-Length", len(msg))
                self.end_headers()
                self.wfile.write(msg)
            else:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.end_headers()

        def do_POST(self):
            """
            Handle an annotate request
            """
            if not self.path.endswith("/"): self.path += "/"
            if self.path == "/annotate/":
                # Read message
                length = int(self.headers.get('content-length'))
                msg = self.rfile.read(length)

                # Do the annotation
                doc = Document()
                parseFromDelimitedString(doc, msg)
                self.annotator.annotate(doc)

                with io.BytesIO() as stream:
                    writeToDelimitedString(doc, stream)
                    msg = stream.getvalue()

                # write message
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/x-protobuf")
                self.send_header("Content-Length", len(msg))
                self.end_headers()
                self.wfile.write(msg)

            else:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.end_headers()

    def __init__(self, host="", port=8432):
        """
        Launches a server endpoint to communicate with CoreNLP
        """
        Process.__init__(self)
        self.host, self.port = host, port
        self._Handler.annotator = self

    def run(self):
        """
        Runs the server using Python's simple HTTPServer.
        TODO: make this multithreaded.
        """
        httpd = HTTPServer((self.host, self.port), self._Handler)
        sa = httpd.socket.getsockname()
        serve_message = "Serving HTTP on {host} port {port} (http://{host}:{port}/) ..."
        print(serve_message.format(host=sa[0], port=sa[1]))
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, exiting.")
            httpd.shutdown()
