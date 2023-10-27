"""
Very short demo for the SceneGraph interface in the CoreNLP server

Requires CoreNLP >= 4.5.5, Stanza >= 1.5.1
"""

import json

from stanza.server import CoreNLPClient

# start_server=None if you have the server running in another process on the same host
# you can start it with whatever normal options CoreNLPClient has
#
# preload=False avoids having the server unnecessarily load annotators
# if you don't plan on using them
with CoreNLPClient(preload=False) as client:
    result = client.scenegraph("Jennifer's antennae are on her head.")
    print(json.dumps(result, indent=2))


