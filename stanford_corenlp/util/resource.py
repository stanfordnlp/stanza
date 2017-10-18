__author__ = 'victor'

import logging
import os
import requests
import stanford_corenlp

def get_from_url(url):
    """
    :param url: url to download from
    :return: return the content at the url
    """
    return requests.get(url).content


def get_data_or_download(dir_name, file_name, url='', size='unknown'):
    """Returns the data. if the data hasn't been downloaded, then first download the data.

    :param dir_name: directory to look in
    :param file_name: file name to retrieve
    :param url: if the file is not found, then download it from this url
    :param size: the expected size
    :return: path to the requested file
    """
    dname = os.path.join(stanford_corenlp.DATA_DIR, dir_name)
    fname = os.path.join(dname, file_name)
    if not os.path.isdir(dname):
        assert url, 'Could not locate data {}, and url was not specified. Cannot retrieve data.'.format(dname)
        os.makedirs(dname)
    if not os.path.isfile(fname):
        assert url, 'Could not locate data {}, and url was not specified. Cannot retrieve data.'.format(fname)
        logging.warn('downloading from {}. This file could potentially be *very* large! Actual size ({})'.format(url, size))
        with open(fname, 'wb') as f:
            f.write(get_from_url(url))
    return fname
