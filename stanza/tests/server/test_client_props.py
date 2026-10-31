"""
Tests for how CoreNLPClient handles the properties argument when
building the server start command.  These do not start a server.
"""

import pytest

from stanza.server.client import CoreNLPClient, StartServer

pytestmark = [pytest.mark.travis, pytest.mark.client]

@pytest.mark.parametrize("language, expected", [
    ("en", "english"),
    ("EN", "english"),
    ("Zh", "chinese"),
    ("english", "english"),
    ("French", "french"),
])
def test_language_properties_case_insensitive(language, expected):
    client = CoreNLPClient(start_server=StartServer.FORCE_START,
                           properties=language,
                           classpath="unused",
                           be_quiet=True)
    assert client.server_props_path == expected
    assert "-serverProperties" in client.start_cmd
    assert client.start_cmd[client.start_cmd.index("-serverProperties") + 1] == expected
