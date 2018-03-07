import rl.pomdp as pomdp
import ply.lex as lex
import ply.yacc as yacc

import pytk.factory as factory
import numpy as np


### LEXER

from rl.parsers.dotpomdp import tokrules as tokrules
lexer = lex.lex(module=tokrules)


### PARSER

class POMDP_Parser:
    tokens = tokrules.tokens

    def __init__(self):
        self.env = None
        self.model = None

        self.S = None
        self.T = None
        self.O = None
        self.R = None

    def p_error(self, p):
        print('Parsing Error:', p.lineno, p.lexpos, p.type, p.value)

    def p_pomdp(self, p):
        """ pomdp : preamble start_state structure_list """

        s0model = pomdp.State0Distribution(self.env)
        s1model = pomdp.State1Distribution(self.env)
        omodel = pomdp.ObsDistribution(self.env)
        rmodel = pomdp.RewardDistribution(self.env)

        s0model.array = self.S
        s1model.array = np.einsum('jik', self.T)
        omodel.array = np.stack(self.env.nstates * [self.O])

        # checking that rewards don't depend on observations..
        if np.any(self.R.mean(axis=-1, keepdims=True) != self.R):
            raise ValueError('I cannot handle rewards which depend on observations')
        rmodel.array = np.einsum('jik', self.R.mean(axis=-1))

        self.env.model = pomdp.Model(self.env, s0model, s1model, omodel, rmodel)
        p[0] = self.env

    def p_preamble(self, p):
        """ preamble : preamble_list """
        # TODO extract stuff from there...
        preamble = dict(p[1])
        gamma = preamble['gamma']
        sfactory = preamble['sfactory']
        afactory = preamble['afactory']
        ofactory = preamble['ofactory']

        env = pomdp.Environment(sfactory, afactory, ofactory)
        env.gamma = gamma

        # better way than this....
        self.T = np.zeros((env.nactions, env.nstates, env.nstates))
        self.O = np.zeros((env.nactions, env.nstates, env.nobs))
        self.R = np.zeros((env.nactions, env.nstates, env.nstates, env.nobs))
        self.env = env

    def p_preamble_list_1(self, p):
        """ preamble_list : preamble_list preamble_item """
        p[0] = p[1] + [p[2]]

    def p_preamble_list_2(self, p):
        """ preamble_list : preamble_item """
        p[0] = [p[1]]

    def p_preamble_item(self, p):
        """ preamble_item : preamble_discount
                          | preamble_value
                          | preamble_state
                          | preamble_action
                          | preamble_obs """
        p[0] = p[1]

    def p_preamble_discount(self, p):
        """ preamble_discount : DISCOUNT COLON FLOAT """
        p[0] = 'gamma', p[3]

    def p_preamble_value(self, p):
        """ preamble_value : VALUES COLON REWARD
                           | VALUES COLON COST """
        if p[3] == 'cost':
            raise ValueError('I do not know how to handle the `cost` keyword')
        p[0] = 'value', p[3]

    def p_preamble_state_1(self, p):
        """ preamble_state : STATES COLON INT """
        p[0] = 'sfactory', factory.FactoryN(p[3])

    def p_preamble_state_2(self, p):
        """ preamble_state : STATES COLON name_list """
        p[0] = 'sfactory', factory.FactoryValues(p[3])

    def p_preamble_action_1(self, p):
        """ preamble_action : ACTIONS COLON INT """
        p[0] = 'afactory', factory.FactoryN(p[3])

    def p_preamble_action_2(self, p):
        """ preamble_action : ACTIONS COLON name_list """
        p[0] = 'afactory', factory.FactoryValues(p[3])

    def p_preamble_obs_1(self, p):
        """ preamble_obs : OBSERVATIONS COLON INT """
        p[0] = 'ofactory', factory.FactoryN(p[3])

    def p_preamble_obs_2(self, p):
        """ preamble_obs : OBSERVATIONS COLON name_list """
        p[0] = 'ofactory', factory.FactoryValues(p[3])

    def p_name_list_1(self, p):
        """ name_list : name_list ID """
        p[0] = p[1] + [p[2]]

    def p_name_list_2(self, p):
        """ name_list : ID """
        p[0] = [p[1]]

    def p_start_state_0(self, p):
        """ start_state : """
        self.S = np.full(self.env.nstates, 1 / self.env.nstates)

    def p_start_state_1_1(self, p):
        """ start_state : START COLON UNIFORM """
        self.S = np.full(self.env.nstates, 1 / self.env.nstates)

    def p_start_state_1_2(self, p):
        """ start_state : START COLON RESET """
        # TODO I suspect this is not supposed to be part of the grammar
        raise ValueError('I do not know how to handle the `reset` keyword')

    # NOTE reduce/reduce conflict solved by enforcing pmatrix contains at least
    # 2 probabilities
    def p_start_state_1_3(self, p):
        """ start_state : START COLON pmatrix """
        pm = p[3]
        self.S = np.array(pm)

    def p_start_state_2(self, p):
        """ start_state : START COLON state """
        s = p[3]
        self.S = np.zeros(self.env.nstates)
        self.S[s] = 1

    def p_start_state_3(self, p):
        """ start_state : START INCLUDE COLON state_list """
        slist = p[4]
        self.S = np.zeros(self.env.nstates)
        self.S[slist] = 1 / len(slist)

    def p_start_state_4(self, p):
        """ start_state : START EXCLUDE COLON state_list """
        slist = p[4]
        self.S = np.full(self.env.nstates, 1 / (self.env.nstates - len(slist)))
        self.S[s0s] = 0

    def p_state_list_1(self, p):
        """ state_list : state_list state """
        p[0] = p[1] + [p[2]]

    def p_state_list_2(self, p):
        """ state_list : state """
        p[0] = [p[1]]

    def p_state_1(self, p):
        """ state : INT """
        p[0] = p[1]

    def p_state_2(self, p):
        """ state : ID """
        p[0] = self.env.sfactory.i(p[1])

    def p_state_3(self, p):
        """ state : ASTERISK """
        p[0] = slice(None)

    def p_action_1(self, p):
        """ action : INT """
        p[0] = p[1]

    def p_action_2(self, p):
        """ action : ID """
        p[0] = self.env.afactory.i(p[1])

    def p_action_3(self, p):
        """ action : ASTERISK """
        p[0] = slice(None)

    def p_obs_1(self, p):
        """ obs : INT """
        p[0] = p[1]

    def p_obs_2(self, p):
        """ obs : ID """
        p[0] = self.env.ofactory.i(p[1])

    def p_obs_3(self, p):
        """ obs : ASTERISK """
        p[0] = slice(None)

    def p_pmatrix_1(self, p):
        """ pmatrix : pmatrix prob """
        p[0] = p[1] + [p[2]]

    # NOTE enforce at least two probabilities;
    # solves reduce/reduce conflict in start_state rule!
    def p_pmatrix_2(self, p):
        """ pmatrix : prob prob """
        p[0] = [p[1], p[2]]

    # def p_pmatrix_2(self, p):
    #     """ pmatrix : prob """
    #     p[0] = [p[1]]

    def p_nmatrix_1(self, p):
        """ nmatrix : nmatrix number """
        p[0] = p[1] + [p[2]]

    def p_nmatrix(self, p):
        """ nmatrix : number """
        p[0] = [p[1]]

    def p_structure_list(self, p):
        """ structure_list : structure_list t_structure
                           | structure_list o_structure
                           | structure_list r_structure
                           | """
        pass

    def p_t_structure_1(self, p):
        """ t_structure : T COLON action COLON state COLON state prob """
        a, s0, s1, pm = p[3], p[5], p[7], p[8]
        self.T[a, s0, s1] = pm

    def p_t_structure_2_1(self, p):
        """ t_structure : T COLON action COLON state UNIFORM """
        a, s0 = p[3], p[5]
        self.T[a, s0] = 1 / self.env.nstates

    def p_t_structure_2_2(self, p):
        """ t_structure : T COLON action COLON state RESET """
        self.T[a, s0] = self.S

    def p_t_structure_2_3(self, p):
        """ t_structure : T COLON action COLON state pmatrix """
        a, s0, pm = p[3], p[5], p[6]
        self.T[a, s0] = pm

    def p_t_structure_3_1(self, p):
        """ t_structure : T COLON action UNIFORM """
        a = p[3]
        self.T[a] = 1 / self.env.nstates

    def p_t_structure_3_2(self, p):
        """ t_structure : T COLON action IDENTITY """
        a = p[3]
        self.T[a] = np.eye(self.env.nstates)

    def p_t_structure_3_3(self, p):
        """ t_structure : T COLON action pmatrix """
        a, pm = p[3], p[4]
        self.T[a] = np.reshape(pm, (self.env.nstates, self.env.nstates))

    def p_o_structure_1(self, p):
        """ o_structure : O COLON action COLON state COLON obs prob """
        a, s1, o, pr = p[3], p[5], p[7], p[8]
        self.O[a, s1, o] = pr

    def p_o_structure_2_1(self, p):
        """ o_structure : O COLON action COLON state UNIFORM """
        a, s1 = p[3], p[5]
        self.O[a, s1] = 1 / self.env.nobs

    def p_o_structure_2_2(self, p):
        """ o_structure : O COLON action COLON state RESET """
        # TODO I suspect this is not supposed to be part of the grammar
        raise ValueError('I do not know how to handle the `reset` keyword')

    def p_o_structure_2_3(self, p):
        """ o_structure : O COLON action COLON state pmatrix """
        a, s1, pm = p[3], p[5], p[6]
        self.O[a, s1] = pm

    def p_o_structure_3_1(self, p):
        """ o_structure : O COLON action UNIFORM """
        a = p[3]
        self.O[a] = 1 / self.env.nobs

    def p_o_structure_3_2(self, p):
        """ o_structure : O COLON action RESET """
        raise ValueError('I do not know how to handle the `reset` keyword')

    def p_o_structure_3_3(self, p):
        """ o_structure : O COLON action pmatrix """
        a, pm = p[3], p[4]
        self.O[a] = np.reshape(pm, (self.env.nstates, self.env.nobs))

    # TODO I could improve this considerably... if I were to not collect
    # nmatrix data in a list... but write to matrix directly!

    def p_r_structure_1(self, p):
        """ r_structure : R COLON action COLON state COLON state COLON obs number """
        a, s0, s1, o, r = p[3], p[5], p[7], p[9], p[10]
        self.R[a, s0, s1, o] = r

    def p_r_structure_2(self, p):
        """ r_structure : R COLON action COLON state COLON state nmatrix """
        a, s0, s1, r = p[3], p[5], p[7], p[8]
        self.R[a, s0, s1] = r

    def p_r_structure_3(self, p):
        """ r_structure : R COLON action COLON state nmatrix """
        a, s0 = p[3], p[5]
        self.R[a, s0] = r

    def p_prob(self, p):
        """ prob : FLOAT
                 | INT """
        p[0] = p[1]

    def p_number_1(self, p):
        """ number : FLOAT
                   | INT """
        p[0] = p[1]

    def p_number_2(self, p):
        """ number : PLUS number
                   | MINUS number """
        p[0] = p[2] if p[1] == '+' else -p[2]


def parse(f, **kwargs):
    p = POMDP_Parser()
    y = yacc.yacc(module=p)
    y.parse(f.read(), lexer=lexer, **kwargs)
    return p.env


from pkg_resources import resource_filename
from contextlib import contextmanager
# import rl


_open = open  # saving default name
@contextmanager
def open(fname):
    try:
        f = _open(fname)
    except FileNotFoundError:
        fname = resource_filename('rl', f'data/pomdp/{fname}.pomdp')
        f = _open(fname)

    yield f
    f.close()

def env(fname):
    with open(fname) as f:
        return parse(f)
