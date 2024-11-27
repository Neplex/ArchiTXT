# Generated from metagrammar.g4 by ANTLR 4.13.2
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
    from typing import TextIO
else:
    from typing.io import TextIO


def serializedATN():
    return [
        4,0,8,81,6,-1,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,
        6,7,6,2,7,7,7,1,0,1,0,1,0,1,0,1,0,1,0,1,0,4,0,25,8,0,11,0,12,0,26,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,38,8,1,11,1,12,1,39,1,2,
        1,2,1,2,1,2,1,2,1,2,1,2,4,2,49,8,2,11,2,12,2,50,1,3,1,3,1,3,1,3,
        1,3,1,3,1,3,1,3,4,3,61,8,3,11,3,12,3,62,1,4,1,4,1,4,1,4,1,4,1,5,
        1,5,1,5,1,6,1,6,1,7,4,7,76,8,7,11,7,12,7,77,1,7,1,7,0,0,8,1,1,3,
        2,5,3,7,4,9,5,11,6,13,7,15,8,1,0,2,3,0,48,57,65,90,97,122,3,0,9,
        10,13,13,32,32,85,0,1,1,0,0,0,0,3,1,0,0,0,0,5,1,0,0,0,0,7,1,0,0,
        0,0,9,1,0,0,0,0,11,1,0,0,0,0,13,1,0,0,0,0,15,1,0,0,0,1,17,1,0,0,
        0,3,28,1,0,0,0,5,41,1,0,0,0,7,52,1,0,0,0,9,64,1,0,0,0,11,69,1,0,
        0,0,13,72,1,0,0,0,15,75,1,0,0,0,17,18,5,82,0,0,18,19,5,69,0,0,19,
        20,5,76,0,0,20,21,5,58,0,0,21,22,5,58,0,0,22,24,1,0,0,0,23,25,7,
        0,0,0,24,23,1,0,0,0,25,26,1,0,0,0,26,24,1,0,0,0,26,27,1,0,0,0,27,
        2,1,0,0,0,28,29,5,71,0,0,29,30,5,82,0,0,30,31,5,79,0,0,31,32,5,85,
        0,0,32,33,5,80,0,0,33,34,5,58,0,0,34,35,5,58,0,0,35,37,1,0,0,0,36,
        38,7,0,0,0,37,36,1,0,0,0,38,39,1,0,0,0,39,37,1,0,0,0,39,40,1,0,0,
        0,40,4,1,0,0,0,41,42,5,69,0,0,42,43,5,78,0,0,43,44,5,84,0,0,44,45,
        5,58,0,0,45,46,5,58,0,0,46,48,1,0,0,0,47,49,7,0,0,0,48,47,1,0,0,
        0,49,50,1,0,0,0,50,48,1,0,0,0,50,51,1,0,0,0,51,6,1,0,0,0,52,53,5,
        67,0,0,53,54,5,79,0,0,54,55,5,76,0,0,55,56,5,76,0,0,56,57,5,58,0,
        0,57,58,5,58,0,0,58,60,1,0,0,0,59,61,7,0,0,0,60,59,1,0,0,0,61,62,
        1,0,0,0,62,60,1,0,0,0,62,63,1,0,0,0,63,8,1,0,0,0,64,65,5,82,0,0,
        65,66,5,79,0,0,66,67,5,79,0,0,67,68,5,84,0,0,68,10,1,0,0,0,69,70,
        5,45,0,0,70,71,5,62,0,0,71,12,1,0,0,0,72,73,5,59,0,0,73,14,1,0,0,
        0,74,76,7,1,0,0,75,74,1,0,0,0,76,77,1,0,0,0,77,75,1,0,0,0,77,78,
        1,0,0,0,78,79,1,0,0,0,79,80,6,7,0,0,80,16,1,0,0,0,6,0,26,39,50,62,
        77,1,6,0,0
    ]

class metagrammarLexer(Lexer):

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    REL = 1
    GROUP = 2
    ENT = 3
    COLL = 4
    ROOT = 5
    PROD_SYMBOL = 6
    PROD_SEPARATOR = 7
    WS = 8

    channelNames = [ u"DEFAULT_TOKEN_CHANNEL", u"HIDDEN" ]

    modeNames = [ "DEFAULT_MODE" ]

    literalNames = [ "<INVALID>",
            "'ROOT'", "'->'", "';'" ]

    symbolicNames = [ "<INVALID>",
            "REL", "GROUP", "ENT", "COLL", "ROOT", "PROD_SYMBOL", "PROD_SEPARATOR",
            "WS" ]

    ruleNames = [ "REL", "GROUP", "ENT", "COLL", "ROOT", "PROD_SYMBOL",
                  "PROD_SEPARATOR", "WS" ]

    grammarFileName = "metagrammar.g4"

    def __init__(self, input=None, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.2")
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None
