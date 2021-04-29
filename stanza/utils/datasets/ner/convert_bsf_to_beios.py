import argparse
import logging
import os
import glob
from collections import namedtuple
import re
from tqdm import tqdm
from random import choices

BsfInfo = namedtuple('BsfInfo', 'id, tag, start_idx, end_idx, token')

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def format_token_as_beios(token: str, tag: str) -> list:
    t_words = token.split()
    res = []
    if len(t_words) == 1:
        res.append(token + ' S-' + tag)
    else:
        res.append(t_words[0] + ' B-' + tag)
        for t_word in t_words[1: -1]:
            res.append(t_word + ' I-' + tag)
        res.append(t_words[-1] + ' E-' + tag)
    return res


def format_token_as_iob(token: str, tag: str) -> list:
    t_words = token.split()
    res = []
    if len(t_words) == 1:
        res.append(token + ' B-' + tag)
    else:
        res.append(t_words[0] + ' B-' + tag)
        for t_word in t_words[1:]:
            res.append(t_word + ' I-' + tag)
    return res


def convert_bsf(data: str, bsf_markup: str, converter: str = 'beios') -> str:
    """
    Convert data file with NER markup in Brat Standoff Format to BEIOS or IOB format.

    :param converter: iob or beios converter to use for document
    :param data: tokenized data to be converted. Each token separated with a space
    :param bsf_markup: Brat Standoff Format markup
    :return: data in BEIOS or IOB format https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)
    """

    def join_simple_chunk(chunk: str) -> list:
        if len(chunk.strip()) == 0:
            return []
        tokens = re.split(r'\s', chunk.strip())
        return [token + ' O' if len(token.strip()) > 0 else token for token in tokens]

    converters = {'beios': format_token_as_beios, 'iob': format_token_as_iob}
    res = []
    markup = parse_bsf(bsf_markup)

    prev_idx = 0
    m_ln: BsfInfo
    for m_ln in markup:
        res += join_simple_chunk(data[prev_idx:m_ln.start_idx])

        convert_f = converters[converter]
        res.extend(convert_f(m_ln.token, m_ln.tag))
        prev_idx = m_ln.end_idx

    if prev_idx < len(data) - 1:
        res += join_simple_chunk(data[prev_idx:])

    return '\n'.join(res)


def parse_bsf(bsf_data: str) -> list:
    """
    Convert textual bsf representation to a list of named entities.

    :param bsf_data: data in the format 'T9	PERS 778 783    токен'
    :return: list of named tuples for each line of the data representing a single named entity token
    """
    if len(bsf_data.strip()) == 0:
        return []

    ln_ptrn = re.compile(r'(T\d+)\s(\w+)\s(\d+)\s(\d+)\s(.+?)(?=T\d+\s\w+\s\d+\s\d+|$)', flags=re.DOTALL)
    result = []
    for m in ln_ptrn.finditer(bsf_data.strip()):
        bsf = BsfInfo(m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), m.group(5).strip())
        result.append(bsf)
    return result


CORPUS_NAME = 'Ukrainian-languk'

def convert_bsf_in_folder(src_dir_path: str, dst_dir_path: str, converter: str = 'beios',
                          doc_delim: str = '\n') -> None:
    """

    :param doc_delim: delimiter to be used between documents
    :param src_dir_path: path to directory with BSF marked files
    :param dst_dir_path: where to save output data
    :param converter: `beios` or `iob` output formats
    :return:
    """
    ann_path = os.path.join(src_dir_path, '*.tok.ann')
    ann_files = glob.glob(ann_path)
    ann_files.sort()

    tok_path = os.path.join(src_dir_path, '*.tok.txt')
    tok_files = glob.glob(tok_path)
    tok_files.sort()

    corpus_folder = os.path.join(dst_dir_path, CORPUS_NAME)
    if not os.path.exists(corpus_folder):
        os.makedirs(corpus_folder)

    if len(ann_files) == 0 or len(tok_files) == 0:
        raise FileNotFoundError(f'Token and annotation files are not found at specified path {ann_path}')
    if len(ann_files) != len(tok_files):
        raise RuntimeError(f'Mismatch between Annotation and Token files. Ann files: {len(ann_files)}, token files: {len(tok_files)}')

    train_set = []
    dev_set = []
    test_set = []

    data_sets = [train_set, dev_set, test_set]
    split_weights = (8, 1, 1)

    log.info(f'Found {len(tok_files)} files')
    for (tok_fname, ann_fname) in tqdm(zip(tok_files, ann_files), total=len(tok_files), unit='file'):
        if tok_fname[:-3] != ann_fname[:-3]:
            tqdm.write(f'Token and Annotation file names do not match ann={ann_fname}, tok={tok_fname}')
            continue

        with open(tok_fname) as tok_file, open(ann_fname) as ann_file:
            token_data = tok_file.read()
            ann_data = ann_file.read()
            out_data = convert_bsf(token_data, ann_data, converter)

            target_dataset = choices(data_sets, split_weights)[0]
            target_dataset.append(out_data)
    log.info(f'Data is split as following: train={len(train_set)}, dev={len(dev_set)}, test={len(test_set)}')

    # writing data to {train/dev/test}.bio files
    names = ['train', 'dev', 'test']
    if doc_delim != '\n':
        doc_delim = '\n' + doc_delim + '\n'
    for idx, name in enumerate(names):
        fname = os.path.join(corpus_folder, name + '.bio')
        with open(fname, 'w') as f:
            f.write(doc_delim.join(data_sets[idx]))
        log.info('Writing to ' + fname)

    log.info('All done')


if __name__ == '__main__':
    logging.basicConfig()

    parser = argparse.ArgumentParser(description='Convert lang-uk NER data set from BSF format to BEIOS format compatible with Stanza NER model training requirements.\n'
                                                 'Original data set should be downloaded from https://github.com/lang-uk/ner-uk\n'
                                                 'For example, create a directory extern_data/lang_uk, then run "git clone git@github.com:lang-uk/ner-uk.git')
    parser.add_argument('--src_dataset', type=str, default='extern_data/ner/lang-uk/ner-uk/data', help='Dir with lang-uk dataset "data" folder (https://github.com/lang-uk/ner-uk)')
    parser.add_argument('--dst', type=str, default='data/ner', help='Where to store the converted dataset')
    parser.add_argument('-c', type=str, default='beios', help='`beios` or `iob` formats to be used for output')
    parser.add_argument('--doc_delim', type=str, default='\n', help='Delimiter to be used to separate documents in the output data')
    parser.print_help()
    args = parser.parse_args()

    convert_bsf_in_folder(args.src_dataset, args.dst, args.c, args.doc_delim)
