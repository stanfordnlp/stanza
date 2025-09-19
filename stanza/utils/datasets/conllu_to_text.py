
import argparse
import re

TEXT_RE = re.compile("^#\\s*text")
NEWPAR_RE = re.compile("^#\\s*newpar")
NEWDOC_RE = re.compile("^#\\s*newdoc")

MWT_RE = re.compile("^\\d+-(\\d+)\t")
WORD_RE = re.compile("^(\\d)+\t")

WORD_NEWPAR_RE = re.compile("NewPar=Yes")
SPACEAFTER_RE = re.compile("SpaceAfter=No")

def print_new_paragraph_if_needed(fout, start, newdoc, newpar, output_buffer):
    if not start and (newdoc or newpar):
        if output_buffer:
            fout.write(output_buffer)
            fout.write("\n")
        fout.write("\n")
        return ""
    return output_buffer

def print_lines_from_buffer(fout, output_buffer, max_len):
    while len(output_buffer) >= max_len:
        split_idx = None
        for idx in range(len(output_buffer)):
            if idx > max_len and split_idx is not None:
                break
            if output_buffer[idx].isspace():
                split_idx = idx
        if split_idx is not None:
            fout.write(output_buffer[:split_idx])
            fout.write("\n")
            output_buffer = output_buffer[split_idx+1:]
        else:
            fout.write(output_buffer)
            fout.write("\n")
            output_buffer = ""
    return output_buffer

def convert_text(conllu_file, output_file):
    with open(conllu_file, encoding="utf-8") as fin:
        lines = fin.readlines()

    with open(output_file, "w", encoding="utf-8") as fout:
        newpar = False
        newdoc = False
        start = True

        in_mwt = False
        mwt_last = None

        def print_and_reset(output_buffer, incoming_buffer):
            nonlocal start, newpar, newdoc, in_mwt

            output_buffer = print_new_paragraph_if_needed(fout, start, newdoc, newpar, output_buffer)
            output_buffer += incoming_buffer
            output_buffer = print_lines_from_buffer(fout, output_buffer, 80)
            start = False
            newpar = False
            newdoc = False
            in_mwt = False
            return output_buffer

        output_buffer = ""
        incoming_buffer = ""

        for line in lines:
            line = line.strip()

            if not line:
                output_buffer = print_and_reset(output_buffer, incoming_buffer)
                incoming_buffer = ""

            if TEXT_RE.match(line):
                # we ignore the #text and extract the text from the tokens
                continue

            if NEWPAR_RE.match(line):
                newpar = True
                continue

            if NEWDOC_RE.match(line):
                newdoc = True
                continue

            match = MWT_RE.match(line)
            if match:
                in_mwt = True
                mwt_last = int(match.group(1))
                pieces = line.split("\t")

                if WORD_NEWPAR_RE.search(pieces[9]):
                    output_buffer = print_and_reset(output_buffer, incoming_buffer)
                    incoming_buffer = ""
                    fout.write(output_buffer)
                    fout.write("\n\n")
                    output_buffer = ""

                incoming_buffer += pieces[1]
                if not SPACEAFTER_RE.search(pieces[9]):
                    incoming_buffer += " "
                continue

            match = WORD_RE.match(line)
            if match:
                pieces = line.split("\t")
                word_id = int(pieces[0])
                if in_mwt and word_id <= mwt_last:
                    continue
                in_mwt = False

                if WORD_NEWPAR_RE.search(pieces[9]):
                    output_buffer = print_and_reset(output_buffer, incoming_buffer)
                    incoming_buffer = ""
                    fout.write(output_buffer)
                    fout.write("\n\n")
                    output_buffer = ""

                incoming_buffer += pieces[1]
                if not SPACEAFTER_RE.search(pieces[9]):
                    incoming_buffer += " "
                continue
        if output_buffer != "":
            fout.write(output_buffer)
            fout.write("\n")

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('conllu_file', type=str, help="CoNLL-U file containing tokens and sentence breaks")
    parser.add_argument('output_file', type=str, help="Plaintext file containing the raw input")
    args = parser.parse_args(args)

    convert_text(args.conllu_file, args.output_file)


if __name__ == "__main__":
    main()
