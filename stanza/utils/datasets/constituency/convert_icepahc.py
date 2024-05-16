from stanza.utils.datasets.constituency import utils

def read_psd_file(input_file):
    """
    Convert the IcePaHC .psd file to text

    Returns a list of sentences
    """
    with open(input_file, encoding='utf-8') as file:
        lines = file.readlines()

    output_trees = []
    current_tree = ''

    # Add the trees as parsed sentences to the output_trees list
    for line in lines:
        if line.startswith("(ROOT"):
            if current_tree:
                cleaned_tree = ' '.join(current_tree.split())
                output_trees.append(cleaned_tree)
            current_tree = line
        else:
            current_tree += line

    # Can't forget the last tree
    if current_tree:
        cleaned_tree = ' '.join(current_tree.split())
        output_trees.append(cleaned_tree.strip())

    return output_trees    


def convert_icepahc_treebank(input_file, train_size=0.8, dev_size=0.1):

    trees = read_psd_file(input_file)

    print("Read %d trees" % len(trees))
    train_trees, dev_trees, test_trees = utils.split_treebank(trees, train_size, dev_size)
    print("Split %d trees into %d train %d dev %d test" % (len(trees), len(train_trees), len(dev_trees), len(test_trees)))

    return train_trees, dev_trees, test_trees


def main():
    treebank = convert_icepahc_treebank("simpleicepahc24.psd")

if __name__ == '__main__':
    main()
