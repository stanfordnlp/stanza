"""
Wrapper functions to run UDPipe modules just as other neural modules. Only one module will be run at each call.

For more information on the UDPipe system, please visit: http://ufal.mff.cuni.cz/udpipe.
"""

import os
import io
import argparse
import subprocess
import time

from stanfordnlp.models.common import conll

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=None, help='Path to input file.')
    parser.add_argument('--output_file', default=None, help='Path to output file.')
    parser.add_argument('--treebank', default=None, help='Full treebank short name.')
    parser.add_argument('--module', choices=['tokenize', 'lemma', 'pos', 'ufeats', 'parse'], help='The module to run at a single step.')
    parser.add_argument('--udpipe_dir', default=None, help='Root directory of UDPipe.')
    parser.add_argument('--short2tb', default='short_to_tb', help='Mapper file from treebank short code to fullname.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    args = vars(args)
    print(f"Running UDPipe with module {args['module']}...")

    # convert names
    short2tb = load_short2tb(args['short2tb'])
    tb_short = args['treebank']
    tb_full = short2tb[tb_short]

    lang_full = tb_full[3:].split('-')[0].lower()
    lang_short, tb_code = tb_short.split('_')

    # look for commands and models
    udpipe_script = f"{args['udpipe_dir']}/bin-linux64/udpipe"
    model_name = f"{lang_full}-{tb_code}-ud-2.2-conll18-180430.udpipe"
    model_file = f"{args['udpipe_dir']}/models/{model_name}"

    if not os.path.exists(model_file):
        model_name = "mixed-ud-ud-2.2-conll18-180430.udpipe"
        model_file = f"{args['udpipe_dir']}/models/{model_name}"

    # check files
    if not args['output_file'].endswith('.conllu'):
        raise Exception("UDPipe module must write to conllu file.")

    if args['module'] == 'tokenize':
        # run tokenizer, ssplit and mwt expander at the same time
        if not args['input_file'].endswith('.txt'):
            raise Exception("UDPipe must take txt file as input when module == tokenize.")
        # run tokenizer from txt file
        udpipe_cmd = f"{udpipe_script} --tokenize {model_file} {args['input_file']} --outfile={args['output_file']} --output=conllu"
        run_udpipe(udpipe_cmd)
        print("Waiting for filesystem...")
        time.sleep(5)
    else:
        if not args['input_file'].endswith('.conllu'):
            raise Exception("UDPipe must take conllu file as input when module != tokenize.")
        # first load the original input file
        input_conll = conll.CoNLLFile(args['input_file'])
        input_conll.load_all()

        # do udpipe
        if args['module'] == 'parse':
            udpipe_cmd = f"{udpipe_script} --parse {model_file} {args['input_file']} --output=conllu --input=conllu"
        else:
            udpipe_cmd = f"{udpipe_script} --tag {model_file} {args['input_file']} --output=conllu --input=conllu"
        udpipe_outputs = run_udpipe(udpipe_cmd, return_stdout=True)
        print("Waiting for filesystem...")
        time.sleep(5)

        # load conll back and merge with original conll
        udpipe_conll = conll.CoNLLFile(input_str=udpipe_outputs.decode())
        udpipe_conll.load_all()
        if args['module'] == 'lemma':
            fields = ['lemma']
        elif args['module'] == 'pos':
            fields = ['upos', 'xpos']
        elif args['module'] == 'ufeats':
            fields = ['feats']
        elif args['module'] == 'parse':
            fields = ['head', 'deprel', 'deps']
        else:
            raise Exception(f"Module {args['module']} not recognized.")

        input_conll.set(fields, udpipe_conll.get(fields)) # set fields back
        # finally write to file
        input_conll.write_conll(args['output_file'])
        print("Waiting for filesystem...")
        time.sleep(5)

    print(f"All done running module {args['module']} with UDPipe.")

def load_short2tb(filename):
    short2tb = dict()
    with open(filename) as infile:
        for line in infile:
            line = line.strip()
            if len(line) == 0:
                continue
            array = line.split()
            assert len(array) == 2
            short2tb[array[0]] = array[1]
    return short2tb

def run_udpipe(cmd, return_stdout=False):
    print(f"Running process: {cmd}")
    if return_stdout:
        rtn = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    else:
        rtn = subprocess.run(cmd, shell=True)
    if rtn.returncode != 0:
        raise Exception(f"Calling UDPipe failed with return code {rtn.returncode}.")
    return rtn.stdout

if __name__ == '__main__':
    main()

