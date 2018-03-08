# import rl.fsc as fsc
import ply.lex as lex
import ply.yacc as yacc

import pytk.factory as factory
import numpy as np


### LEXER

from rl.parsers.dotfsc import tokrules as tokrules
lexer = lex.lex(module=tokrules)


### PARSER

class FSC_Parser:
    tokens = tokrules.tokens

    def __init__(self):
        self.A = None
        self.N = None

    def p_error(self, p):
        print('Parsing Error:', p.lineno, p.lexpos, p.type, p.value)

    def p_fsc(self, p):
        """ fsc : preamble start_node structure_list """
        # TODO checking that nodes are all connected
        # TODO checking that all actions are feasible
        self.amask = self.A.T
        self.nmask = self.N.T

    def p_preamble(self, p):
        """ preamble : preamble_list """
        preamble = dict(p[1])
        self.nfactory = preamble['nfactory']
        self.afactory = preamble['afactory']

        self.A = np.ones((self.nfactory.nitems, self.afactory.nitems), dtype=bool)
        self.N = np.ones((self.nfactory.nitems, self.nfactory.nitems), dtype=bool)

    def p_preamble_list_1(self, p):
        """ preamble_list : preamble_list preamble_item """
        p[0] = p[1] + [p[2]]

    def p_preamble_list_2(self, p):
        """ preamble_list : preamble_item """
        p[0] = [p[1]]

    def p_preamble_item(self, p):
        """ preamble_item : preamble_node
                          | preamble_action """
        p[0] = p[1]

    def p_preamble_node_1(self, p):
        """ preamble_node : NODES COLON INT """
        p[0] = 'nfactory', factory.FactoryN(p[3])

    def p_preamble_node_2(self, p):
        """ preamble_node : NODES COLON name_list """
        p[0] = 'nfactory', factory.FactoryValues(p[3])

    def p_preamble_action_1(self, p):
        """ preamble_action : ACTIONS COLON INT """
        p[0] = 'afactory', factory.FactoryN(p[3])

    def p_preamble_action_2(self, p):
        """ preamble_action : ACTIONS COLON name_list """
        p[0] = 'afactory', factory.FactoryValues(p[3])

    def p_name_list_1(self, p):
        """ name_list : name_list ID """
        p[0] = p[1] + [p[2]]

    def p_name_list_2(self, p):
        """ name_list : ID """
        p[0] = [p[1]]

    def p_start_node_0(self, p):
        """ start_node : """
        self.start_node = 0

    def p_start_node_1(self, p):
        """ start_node : START COLON node """
        n = p[3]
        self.start_node = n

    def p_node_1(self, p):
        """ node : INT """
        p[0] = p[1]

    def p_node_2(self, p):
        """ node : ID """
        p[0] = self.nfactory.i(p[1])

    def p_node_3(self, p):
        """ node : ASTERISK """
        p[0] = slice(None)

    def p_action_1(self, p):
        """ action : INT """
        p[0] = p[1]

    def p_action_2(self, p):
        """ action : ID """
        p[0] = self.afactory.i(p[1])

    def p_action_3(self, p):
        """ action : ASTERISK """
        p[0] = slice(None)

    def p_structure_list(self, p):
        """ structure_list : structure_list a_structure
                           | structure_list n_structure
                           | """
        pass

    def p_a_structure_1(self, p):
        """ a_structure : A COLON node COLON action bool """
        n, a, b = p[3], p[5], p[6]
        self.A[n, a] = b

    def p_a_structure_2(self, p):
        """ a_structure : A COLON node bmatrix """
        n, bm = p[3], p[4]
        self.A[n] = bm

    def p_a_structure_3(self, p):
        """ a_structure : A COLON node ALL """
        n = p[3]
        self.A[n].fill(True)

    def p_a_structure_4(self, p):
        """ a_structure : A COLON node NONE """
        n = p[3]
        self.A[n].fill(False)

    # def p_a_structure_5(self, p):
    #     """ a_structure : A COLON bmatrix """
    #     bm = p[3]
    #     self.A = np.reshape(bm, (self.nfactory.nitems, self.afactory.nitems))

    def p_a_structure_6(self, p):
        """ a_structure : A COLON ALL """
        self.A.fill(True)

    def p_a_structure_7(self, p):
        """ a_structure : A COLON NONE """
        self.A.fill(False)

    def p_n_structure_1(self, p):
        """ n_structure : N COLON node COLON node bool """
        n, n1, b = p[3], p[5], p[6]
        self.N[n, n1] = b

    def p_n_structure_2(self, p):
        """ n_structure : N COLON node bmatrix """
        n, bm = p[3], p[4]
        self.N[n] = bm

    def p_n_structure_3(self, p):
        """ n_structure : N COLON node ALL """
        n = p[3]
        self.N[n].fill(True)

    def p_n_structure_4(self, p):
        """ n_structure : N COLON node NONE """
        n = p[3]
        self.N[n].fill(False)

    # def p_n_structure_5(self, p):
    #     """ n_structure : N COLON bmatrix """
    #     bm = p[3]
    #     self.N = np.reshape(bm, (self.nfactory.nitems, self.nfactory.nitems))

    def p_n_structure_6(self, p):
        """ n_structure : N COLON ALL """
        self.N.fill(True)

    def p_n_structure_7(self, p):
        """ n_structure : N COLON NONE """
        self.N.fill(False)

    def p_bmatrix_1(self, p):
        """ bmatrix : bmatrix bool """
        p[0] = p[1] + [p[2]]

    def p_bmatrix_2(self, p):
        """ bmatrix : bool """
        p[0] = [p[1]]

    def p_bool(self, p):
        """ bool : INT """
        p[0] = bool(p[1])


def parse(f, **kwargs):
    p = FSC_Parser()
    y = yacc.yacc(module=p)
    y.parse(f.read(), lexer=lexer, **kwargs)
    return p.start_node, p.amask, p.nmask


from pkg_resources import resource_filename
from contextlib import contextmanager
# import rl


_open = open  # saving default name
@contextmanager
def open(fname):
    try:
        f = _open(f'{fname}.fsc')
    except FileNotFoundError:
        fname = resource_filename('rl', f'data/fsc/{fname}.fsc')
        f = _open(fname)

    yield f
    f.close()


import rl.pomdp.policies as policies

def fsc(fname, env):
    with open(fname) as f:
        n0, amask, nmask = parse(f)
    return policies.StructuredFSC(env, amask, nmask, n0)
