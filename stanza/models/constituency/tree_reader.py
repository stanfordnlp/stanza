from sly import Lexer, Parser

from stanza.models.constituency.parse_tree import Tree

class TreeLexer(Lexer):
    tokens = { LPAREN, RPAREN, TOKEN }

    # String containing ignored characters (between tokens)
    ignore = ' \t'

    LPAREN = r'\('
    RPAREN = r'\)'

    TOKEN = r'[^()\n ]+'

    # Define a rule so we can track line numbers
    @_(r'\n+')
    def ignore_newline(self, t):
        self.lineno += len(t.value)

class TreeParser(Parser):
    tokens = TreeLexer.tokens

    # TODO: hopefully there's some other way to parse multiple trees from one file
    @_('root treelist')
    def treelist(self, p):
      trees = p.treelist
      trees.append(p.root)
      return trees

    @_('root')
    def treelist(self, p):
        return [p.root]

    # the extra layer of productions at the top is so that we can handle trees such as
    #  ((tree stuff))
    # by adding a ROOT node at the very top
    @_('LPAREN factor RPAREN')
    def root(self, p):
        return Tree(label="ROOT", children=[p.factor])

    @_('factor')
    def root(self, p):
        return p.factor

    @_('LPAREN TOKEN subtrees RPAREN')
    def factor(self, p):
        token = p.TOKEN
        children = p.subtrees
        return Tree(label=token, children=children)

    @_('LPAREN TOKEN words RPAREN')
    def factor(self, p):
        token = p.TOKEN
        text = p.words
        # We join the text into one token
        # This is relevant in the case of VI, for example,
        # or the FR token with a space in the middle of a number
        text = " ".join(text)
        child = Tree(label=text)
        return Tree(label=token, children=[child])

    @_('LPAREN TOKEN RPAREN')
    def factor(self, p):
        return Tree(label=p.TOKEN)

    @_('factor')
    def subtrees(self, p):
        return [p.factor]

    @_('subtrees factor')
    def subtrees(self, p):
        children = p.subtrees
        children.append(p.factor)
        return children

    @_('TOKEN')
    def words(self, p):
        return [p.TOKEN]

    @_('words TOKEN')
    def words(self, p):
        words = p.append(p.TOKEN)
        return words

def read_trees(text):
    """
    Reads multiple trees from the text

    TODO: there needs to be some error handling to recover from cases such as too many )
    or to alert for broken trees and which line they occur on
    and we should check for the result of the reading
    """
    lexer = TreeLexer()
    parser = TreeParser()
    trees = parser.parse(lexer.tokenize(text))
    trees.reverse()
    return trees

def read_tree_file(filename):
    with open(filename) as fin:
        trees = read_trees(fin.read())
    return trees

if __name__ == '__main__':
    text="( (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))"
    trees = read_trees(text)
    print(trees)
