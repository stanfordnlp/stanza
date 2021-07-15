from collections import defaultdict
import pickle
from conllu import parse_incr
class Trie:
    """
    A simple Trie with add, search, and startsWith functions.
    """
    def __init__(self):
        self.root = defaultdict()

    def add(self, word):
        current = self.root
        for letter in word:
            current = current.setdefault(letter, {})
        current.setdefault("_end")

    def search(self, word):
        current = self.root
        for letter in word:
            if letter not in current:
                return False
            current = current[letter]
        if "_end" in current:
            return True
        return False

    def startsWith(self, prefix):
        current = self.root
        for letter in prefix:
            if letter not in current:
                return False
            current = current[letter]
        return True

def main(lang, train_path, external_path, dict_path):
    tree = Trie()
    word_list = set()
    if train_path!=None:
        train_file = open(train_path, "r", encoding="utf-8")
        for tokenlist in parse_incr(train_file):
            for token in tokenlist:
                word = token['form']
                word = word.lower()
                #check multiple_syllable word for vi
                if lang == "vi_vlsp":
                    if len(word.split(" ")) > 1:
                        #do not include the words that includes numbers.
                        if not any(map(str.isdigit, word)):
                            tree.add(word)
                            word_list.add(word)
                else:
                    if len(word)>1:
                        if not any(map(str.isdigit, word)):
                            tree.add(word)
                            word_list.add(word)
        print("Added ", len(word_list), " words found in training set to dictionary.")
    if external_path != None:
        external_file = open(external_path, "r", encoding="utf-8")
        lines = external_file.readlines()
        for line in lines:
            word = line.lower()
            word = word.replace("\n","")
            # check multiple_syllable word for vi
            if lang == "vi_vlsp":
                if len(word.split(" ")) > 1:
                    if not any(map(str.isdigit, word)):
                        tree.add(word)
                        word_list.add(word)
            else:
                if len(word)>1:
                    if not any(map(str.isdigit, word)):
                        tree.add(word)
                        word_list.add(word)

    if len(word_list)>0:
        with open(dict_path, 'wb') as config_dictionary_file:
            pickle.dump(tree, config_dictionary_file)
        config_dictionary_file.close()
        print("Succesfully generated dict file with total of ", len(word_list), " words.")

if __name__=='__main__':
    main()
