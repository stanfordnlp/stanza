"""
Very short demo for the SceneGraph interface in the CoreNLP server

Requires CoreNLP >= 4.5.5, Stanza >= 1.5.1
"""

from stanza.server import CoreNLPClient

# start_server=None is if you have the server running in another process on the same host
# you can start it with whatever normal options CoreNLPClient has
with CoreNLPClient(start_server=None) as client:
    result = client.scenegraph("Jennifer's antennae are on her head.")
    print(result)


