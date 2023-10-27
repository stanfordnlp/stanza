"""
Test the utils file of the NER dataset processing
"""

import pytest

from stanza.utils.datasets.ner.utils import list_doc_entities
from stanza.tests.datasets.ner.test_prepare_ner_file import BIO_1, BIO_2, write_and_convert

def test_list_doc_entities(tmp_path):
    """
    Test the function which lists all of the entities in a doc
    """
    doc = write_and_convert(tmp_path, BIO_1)
    entities = list_doc_entities(doc)
    expected = [(('Jennifer', "Sh'reyan"), 'PERSON')]
    assert expected == entities

    doc = write_and_convert(tmp_path, BIO_2)
    entities = list_doc_entities(doc)
    expected = [(('Jennifer',), 'PERSON'), (('Beckett',), 'PERSON'), (('Cerritos',), 'LOCATION')]
    assert expected == entities    

    doc = write_and_convert(tmp_path, "\n\n".join([BIO_1, BIO_2]))
    entities = list_doc_entities(doc)
    expected = [(('Jennifer', "Sh'reyan"), 'PERSON'), (('Jennifer',), 'PERSON'), (('Beckett',), 'PERSON'), (('Cerritos',), 'LOCATION')]
    assert expected == entities

    doc = write_and_convert(tmp_path, "\n\n".join([BIO_1, BIO_1, BIO_2]))
    entities = list_doc_entities(doc)
    expected = [(('Jennifer', "Sh'reyan"), 'PERSON'), (('Jennifer', "Sh'reyan"), 'PERSON'), (('Jennifer',), 'PERSON'), (('Beckett',), 'PERSON'), (('Cerritos',), 'LOCATION')]
    assert expected == entities


