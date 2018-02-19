import rl.pomdp as pomdp
import ply.lex as lex
import ply.yacc as yacc

import pytk.factory as factory
import numpy as np


tokens = (
    'COLON',
    'ASTERISK',
    'PLUS',
    'MINUS',
    'STRING',
    'INT',
    'FLOAT',
)

reserved = {
    'discount': 'DISCOUNT',
    'values': 'VALUES',
    'states': 'STATES',
    'actions': 'ACTIONS',
    'observations': 'OBSERVATIONS',
    'T': 'T',
    'O': 'O',
    'R': 'R',
    'uniform': 'UNIFORM',
    'identity': 'IDENTITY',
    'reward': 'REWARD',
    'cost': 'COST',
    'start': 'START',
    'include': 'INCLUDE',
    'exclude': 'EXCLUDE',
    'reset': 'RESET',
}

tokens += tuple(reserved.values())

t_COLON = r':'
t_ASTERISK = r'\*'
t_PLUS = r'\+'
t_MINUS = r'-'

def t_STRING(t):
    r'[a-zA-Z][a-zA-Z0-9\_\-]*'
    if t.value in reserved:
        t.type = reserved[t.value]
    return t


def t_NUMBER(t):
    r'[0-9]*\.?[0-9]+((E|e)(\+|-)?[0-9]+)?'
    try:
        t.value = int(t.value)
    except ValueError:
        pass
    else:
        t.type = 'INT'
        return t

    try:
        t.value = float(t.value)
    except ValueError:
        pass
    else:
        t.type = 'FLOAT'
        return t

def t_COMMENT(t):
    r'\#.*'
    pass

t_ignore = ' \t'

# updates line number
def t_newline(t):
    r'\n'
    t.lexer.lineno += 1

def t_error(t):
    print(f'Illegal character \'{t.value[0]}\'')
    t.lexer.skip(1)

lex.lex()


class POMDP_Parser:
    tokens = tokens

    def __init__(self):
        self.env = None
        self.model = None

        self.S = None
        self.T = None
        self.O = None
        self.R = None

    def p_error(self, p):
        print('Parsing Error:', p.lineno, p.lexpos, p.type, p.value)

    def p_pomdp_file(self, p):
        """ pomdp_file : preamble start_state param_list """

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

    def p_preamble_list(self, p):
        """ preamble_list : preamble_list preamble_item
                          | preamble_item """
        try:
            p[0] = p[1] + [p[2]]
        except IndexError:
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
        p[0] = 'value', p[3]

    def p_preamble_state(self, p):
        """ preamble_state : STATES COLON INT
                           | STATES COLON name_list """
        if isinstance(p[3], int):
            sfactory = factory.FactoryN(p[3])
        else:
            sfactory = factory.FactoryValues(p[3])
        p[0] = 'sfactory', sfactory

    def p_preamble_action(self, p):
        """ preamble_action : ACTIONS COLON INT
                            | ACTIONS COLON name_list """
        if isinstance(p[3], int):
            afactory = factory.FactoryN(p[3])
        else:
            afactory = factory.FactoryValues(p[3])
        p[0] = 'afactory', afactory

    def p_preamble_obs(self, p):
        """ preamble_obs : OBSERVATIONS COLON INT
                         | OBSERVATIONS COLON name_list """
        if isinstance(p[3], int):
            ofactory = factory.FactoryN(p[3])
        else:
            ofactory = factory.FactoryValues(p[3])
        p[0] = 'ofactory', ofactory

    def p_name_list(self, p):
        """ name_list : name_list STRING
                      | STRING """
        try:
            p[0] = p[1] + [p[2]]
        except IndexError:
            p[0] = [p[1]]

    def p_start_state(self, p):
        """ start_state : START COLON umatrix
                        | START COLON STRING
                        | START INCLUDE COLON start_state_list
                        | START EXCLUDE COLON start_state_list
                        | """
        if len(p) == 4:
            pr = p[3]
            if pr == 'uniform':
                self.S = np.full((self.env.nstates,), 1 / self.env.nstates)
            elif pr == 'reset':
                raise ValueError
            else:
                self.S = np.array(pr)
        elif len(p) == 6:
            s0s = p[4]
            if p[2] == 'include':
                self.S = np.zeros((self.env.nstates,))
                self.S[s0s] = 1 / len(s0s)
            else:  # exclude
                self.S = np.full((self.env.nstates,), 1 / (self.env.nstates - len(s0s)))
                self.S[s0s] = 0

    def p_start_state_list(self, p):
        """ start_state_list : start_state_list state
                             | state """
        try:
            p[0] = p[1] + [p[2]]
        except IndexError:
            p[0] = [p[1]]

    def p_state(self, p):
        """ state : INT
                  | STRING
                  | ASTERISK """
        if p[1] == '*':
            p[0] = slice(None)
        elif isinstance(p[1], int):
            p[0] = p[1]
        else:
            p[0] = self.env.sfactory.i(p[1])

    def p_action(self, p):
        """ action : INT
                   | STRING
                   | ASTERISK """
        if p[1] == '*':
            p[0] = slice(None)
        elif isinstance(p[1], int):
            p[0] = p[1]
        else:
            p[0] = self.env.afactory.i(p[1])

    def p_obs(self, p):
        """ obs : INT
                | STRING
                | ASTERISK """
        if p[1] == '*':
            p[0] = slice(None)
        elif isinstance(p[1], int):
            p[0] = p[1]
        else:
            p[0] = self.env.ofactory.i(p[1])

    def p_umatrix(self, p):
        """ umatrix : UNIFORM
                    | RESET
                    | pmatrix """
        p[0] = p[1]

    def p_uimatrix(self, p):
        """ uimatrix : UNIFORM
                     | IDENTITY
                     | pmatrix """
        p[0] = p[1]

    def p_pmatrix(self, p):
        """ pmatrix : pmatrix prob
                    | prob """
        try:
            p[0] = p[1] + [p[2]]
        except IndexError:
            p[0] = [p[1]]

    def p_nmatrix(self, p):
        """ nmatrix : nmatrix number
                    | number """
        try:
            p[0] = p[1] + [p[2]]
        except IndexError:
            p[0] = [p[1]]

    def p_param_list(self, p):
        """ param_list : param_list t_spec
                       | param_list o_spec
                       | param_list r_spec
                       | """
        pass

    def p_t_spec(self, p):
        """ t_spec : T COLON action COLON state COLON state prob
                   | T COLON action COLON state umatrix
                   | T COLON action uimatrix """
        # TODO it's not so nice that I have to check length stuff here...
        a = p[3]
        if len(p) == 9:
            s0 = p[5]
            s1 = p[7]
            pr = p[8]
            self.T[a, s0, s1] = pr
        elif len(p) == 7:
            s0 = p[5]
            pr = p[6]
            if pr == 'uniform':
                self.T[a, s0] = 1 / self.env.nstates
            elif pr == 'reset':
                raise ValueError
            else:
                self.T[a, s0] = pr
        else:
            pr = p[4]
            if pr == 'uniform':
                self.T[a] = 1 / self.env.nstates
            elif pr == 'identity':
                self.T[a] = np.eye(self.env.nstates)
            else:
                self.T[a] = np.reshape(pr, self.T[a].shape)

    def p_o_spec(self, p):
        """ o_spec : O COLON action COLON state COLON obs prob
                   | O COLON action COLON state umatrix
                   | O COLON action umatrix """
        a = p[3]
        if len(p) == 9:
            s1 = p[5]
            o = p[7]
            pr = p[8]
            self.O[a, s1, o] = pr
        elif len(p) == 7:
            s1 = p[5]
            pr = p[6]
            if pr == 'uniform':
                self.O[a, s1] = 1 / self.env.nobs
            elif pr == 'reset':
                raise ValueError
            else:
                self.O[a, s1] = pr
        else:
            pr = p[4]
            if pr == 'uniform':
                self.O[a] = 1 / self.env.nstates
            elif pr == 'reset':
                raise ValueError
            else:
                self.O[a] = np.reshape(pr, self.O[a].shape)

    # TODO I could improve this considerably... if I were to not collect
    # nmatrix data in a list... but write to matrix directly!

    def p_r_spec(self, p):
        """ r_spec : R COLON action COLON state COLON state COLON obs number
                   | R COLON action COLON state COLON state nmatrix
                   | R COLON action COLON state nmatrix """
        a = p[3]
        s0 = p[5]
        if len(p) == 11:
            s1 = p[7]
            o = p[9]
            r = p[10]
            self.R[a, s0, s1, o] = r
        elif len(p) == 9:
            s1 = p[7]
            r = p[8]
            self.R[a, s0, s1] = r
        else:
            self.R[a, s0] = r

    def p_prob(self, p):
        """ prob : FLOAT
                 | INT """
        p[0] = p[1]

    def p_number(self, p):
        """ number : optional_sign FLOAT
                   | optional_sign INT """
        if p[1] == '+':
            p[0] = p[2]
        else:
            p[0] = -p[2]

    def p_optional_sign(self, p):
        """ optional_sign : PLUS
                          | MINUS
                          | """
        if len(p) > 1 and p[1] == '-':
            p[0] = '-'
        else:
            p[0] = '+'


def parse(f):
    p = POMDP_Parser()
    y = yacc.yacc(module=p)
    y.parse(f.read())
    return p.env
