from ctypes.wintypes import PBYTE
import sys
from synth.specification import PBE

from synth.task import Dataset
from synth.semantic import DSLEvaluator
from synth.syntax import DSL, PrimitiveType, Arrow, List, INT, STRING


import string
import re
from regexp.type_regex import regex_match, Raw, REGEXP

from synth.syntax.type_system import BOOL

init = PrimitiveType("")
generalized_to_re = {
    "U": "[A-Z]",
    "L": "[a-z]",
    "N": "[0-9]",
    "O": "^[\d\A\a]",
    "W": " ",
    "begin": ""
}

#### Make generalized primitive types
#### Make method that generates dataset
#### -> tasks should be: input: str, output: bool -> "06 56 85 69 93", True / "0a 56 8a 69 9x", False

# intensification: si on sait faire 3 exemples, alors la structure du programme peut donner pour 5 exemples
# c'est à dire que si on a un arbre qui marche pas trop mal, on explore ses environs

## question: comment calculer la probabilité qu'un mot soit trouvé pour une regexp?
# faire un automate pour chaque regexp trouvée, le déterminiser, faire de nombreux runs dessus. Chaque run génère donc un mot
# On fait un set de tous les runs, et on considère une probabilité uniforme dessus.

# prendre un token de base (vide, genre \b), cf t0 dans deepcoder/dreamcoder
# le faire passer dans des méthodes qui l'étendent à chaque fois d'un caractère (on concat des strings en gros pour trouver la solution)

def __qmark__(x): return x + '?' 

def __kleene__(x): return x + '*'

def __plus__(x): return x + '+'

def __numbered__(x): return lambda n: x + r"{n}"

def __lowercase__(x): return x + 'L'

def __uppercase__(x): return x + 'U'

def __number__(x): return x + 'N'

def __other__(x): return x + 'O'

def __whitespace__(x): return x + 'W'

# def __alt__(x): return lambda y: x + '|' + y  

# def __min__(x): return lambda n: x + ''

def __eval__(x, reg): 
    modified = ''
    for char in reg:
        if char in generalized_to_re:
            modified += generalized_to_re[char]
        else:
            modified += char
    x = ''.join(x)
    result = regex_match(Raw(modified), x)
    #print(f"{result.match.group() if result else None} vs {x} => {result.string == x if result != None else False}")
    if result is None:
        return False
    return result.match.group() == x

"""
def __alt__(x, y): pass # x|y (one of the two)

def __min__(x, mi): pass #x{mi,} (repeated at least mi times)

def __max__(x, ma): pass #x{,ma} (repeated up to ma times)

def __borned__(x, mi, ma): pass #x{mi,ma} (repeated between mi and ma times)
"""
#U,N,PLUS,O,N,U,U,
#print(__uppercase__(__uppercase__(__number__(__whitespace__(__plus__(__number__(__uppercase__(init.type_name))))))))
#print('('.join(["U", "U", "N", " ", "PLUS", "N", "U", "begin"]))
#U,U,N, ,PLUS,N,U,begin
# list to be taken from re (https://docs.python.org/3/library/re.html), used here as reference
__semantics = {
    "begin": init.type_name,
    "?": __qmark__,
    "*": __kleene__,
    "+": __plus__,
    "num": __numbered__,
    "U": __uppercase__,
    "L": __lowercase__,
    "N": __number__,
    "O": __other__,
    "W": __whitespace__,
    "eval": lambda x: lambda reg: __eval__(x, reg),
   # "|": __alt__,
   # "min": __min__,
   # "max": __max__,
   # "borned": __borned__
}

__primitive_types = {
    "begin": REGEXP,
    "?": Arrow(REGEXP, REGEXP),
    "*": Arrow(REGEXP, REGEXP),
    "+": Arrow(REGEXP, REGEXP),
    "num": Arrow(REGEXP, Arrow(INT, REGEXP)), # rajouter des int
    "U": Arrow(REGEXP, REGEXP),
    "L": Arrow(REGEXP, REGEXP),
    "N": Arrow(REGEXP, REGEXP),
    "O": Arrow(REGEXP, REGEXP),
    "W": Arrow(REGEXP, REGEXP),
    "eval": Arrow(List(STRING), Arrow(REGEXP, BOOL)),
   # "|": Arrow(CHAR, Arrow(CHAR, CHAR)),
   # "min": Arrow(CHAR, Arrow(INT, List[CHAR])),
   # "max": Arrow(CHAR, Arrow(INT, List[CHAR])),
   # "borned": Arrow(CHAR, Arrow(INT, Arrow(INT, List[CHAR])))
   ## séparer différents types de constantes:
   ### spécialisées (ex: 'A', "Dr. ")
   ### généralisées (ex: tous les caractères minuscules)
}

dsl = DSL(__primitive_types)
evaluator = DSLEvaluator(__semantics)
evaluator.skip_exceptions.add(re.error)
lexicon  = list([chr(i) for i in range(32,126)])

