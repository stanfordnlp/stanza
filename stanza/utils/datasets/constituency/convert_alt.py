"""
Read files of parses and the files which define the train/dev/test splits

Write out the files after splitting them

Sequence of operations:
  - read the raw lines from the input files
  - read the recommended splits, as per the ALT description page
  - separate the trees using the recommended split files
  - write back the trees
"""

def read_split_file(split_file):
    """
    Read a split file for ALT

    The format of the file is expected to be a list of lines such as
    URL.1234    <url>
    Here, we only care about the id

    return: a set of the ids
    """
    with open(split_file, encoding="utf-8") as fin:
        lines = fin.readlines()
    lines = [x.strip() for x in lines]
    lines = [x.split()[0] for x in lines if x]
    if any(not x.startswith("URL.") for x in lines):
        raise ValueError("Unexpected line in %s: %s" % (split_file, x))
    split = set(int(x.split(".", 1)[1]) for x in lines)
    return split

def split_trees(all_lines, splits):
    """
    Splits lines of the form
    SNT.17873.4049	(S ...
    then assigns them to a list based on the file id in
    SNT.<file>.<sent>
    """
    trees = [list() for _ in splits]
    for line in all_lines:
        tree_id, tree_text = line.split(maxsplit=1)
        tree_id = int(tree_id.split(".", 2)[1])
        for split_idx, split in enumerate(splits):
            if tree_id in split:
                trees[split_idx].append(tree_text)
                break
        else:
            # couldn't figure out which split to put this in
            raise ValueError("Couldn't find which split this line goes in:\n%s" % line)
    return trees

def read_alt_lines(input_files):
    """
    Read the trees from the given file(s)

    Any trees with wide spaces are eliminated.  The parse tree
    handling doesn't handle it well and the tokenizer won't produce
    tokens which are entirely wide spaces anyway

    The tree lines are not processed into trees, though
    """
    all_lines = []
    for input_file in input_files:
        with open(input_file, encoding="utf-8") as fin:
            all_lines.extend(fin.readlines())
    all_lines = [x.strip() for x in all_lines]
    all_lines = [x for x in all_lines if x]
    original_count = len(all_lines)
    # there is 1 tree with wide space as an entire token, and 4 with wide spaces at the end of a token
    all_lines = [x for x in all_lines if not "　" in x]
    new_count = len(all_lines)
    if new_count < original_count:
        print("Eliminated %d trees for having wide spaces in it" % ((original_count - new_count)))
        original_count = new_count
    all_lines = [x for x in all_lines if not "\\x" in x]
    new_count = len(all_lines)
    if new_count < original_count:
        print("Eliminated %d trees for not being correctly encoded" % ((original_count - new_count)))
        original_count = new_count
    return all_lines

def convert_alt(input_files, split_files, output_files):
    """
    Convert the ALT treebank into train/dev/test splits

    input_files: paths to read trees
    split_files: recommended splits from the ALT page
    output_files: where to write train/dev/test
    """
    all_lines = read_alt_lines(input_files)

    splits = [read_split_file(split_file) for split_file in split_files]
    trees = split_trees(all_lines, splits)

    for chunk, output_file in zip(trees, output_files):
        print("Writing %d trees to %s" % (len(chunk), output_file))
        with open(output_file, "w", encoding="utf-8") as fout:
            for tree in chunk:
                # the extra ROOT is because the ALT doesn't have this at the top of its trees
                fout.write("(ROOT {})\n".format(tree))
