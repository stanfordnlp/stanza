
import glob
import os

def convert_nytk(base_input_path, base_output_path, short_name):
    for shard in ('train', 'dev', 'test'):
        if shard == 'dev':
            base_input_subdir = os.path.join(base_input_path, "data/train-devel-test/devel")
        else:
            base_input_subdir = os.path.join(base_input_path, "data/train-devel-test", shard)

        shard_lines = []
        base_input_glob = base_input_subdir + "/*/no-morph/*"
        subpaths = glob.glob(base_input_glob)
        print("Reading %d input files from %s" % (len(subpaths), base_input_glob))
        for input_filename in subpaths:
            if len(shard_lines) > 0:
                shard_lines.append("")
            with open(input_filename) as fin:
                lines = fin.readlines()
                if lines[0].strip() != '# global.columns = FORM LEMMA UPOS XPOS FEATS CONLL:NER':
                    raise ValueError("Unexpected format in %s" % input_filename)
                lines = [x.strip().split("\t") for x in lines[1:]]
                lines = ["%s\t%s" % (x[0], x[5]) if len(x) > 1 else "" for x in lines]
                shard_lines.extend(lines)

        bio_filename = os.path.join(base_output_path, '%s.%s.bio' % (short_name, shard))
        with open(bio_filename, "w") as fout:
            print("Writing %d lines to %s" % (len(shard_lines), bio_filename))
            for line in shard_lines:
                fout.write(line)
                fout.write("\n")
