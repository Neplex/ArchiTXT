# Generated from metagrammar.g4 by ANTLR 4.13.2
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,8,92,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,6,
        2,7,7,7,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,38,8,1,1,2,1,2,1,2,1,2,1,2,1,2,1,
        2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,3,2,61,
        8,2,1,3,1,3,1,3,1,3,1,3,1,4,1,4,1,4,1,4,1,4,1,5,1,5,1,5,1,5,1,5,
        1,5,1,6,1,6,1,6,1,6,1,6,1,7,1,7,1,7,1,7,1,7,1,7,3,7,90,8,7,1,7,0,
        0,8,0,2,4,6,8,10,12,14,0,0,91,0,16,1,0,0,0,2,37,1,0,0,0,4,60,1,0,
        0,0,6,62,1,0,0,0,8,67,1,0,0,0,10,72,1,0,0,0,12,78,1,0,0,0,14,89,
        1,0,0,0,16,17,5,5,0,0,17,18,5,6,0,0,18,19,3,2,1,0,19,20,5,7,0,0,
        20,21,3,4,2,0,21,22,5,0,0,1,22,23,6,0,-1,0,23,1,1,0,0,0,24,38,6,
        1,-1,0,25,26,5,2,0,0,26,27,3,2,1,0,27,28,6,1,-1,0,28,38,1,0,0,0,
        29,30,5,1,0,0,30,31,3,2,1,0,31,32,6,1,-1,0,32,38,1,0,0,0,33,34,5,
        4,0,0,34,35,3,2,1,0,35,36,6,1,-1,0,36,38,1,0,0,0,37,24,1,0,0,0,37,
        25,1,0,0,0,37,29,1,0,0,0,37,33,1,0,0,0,38,3,1,0,0,0,39,61,6,2,-1,
        0,40,41,3,6,3,0,41,42,5,7,0,0,42,43,3,4,2,0,43,44,6,2,-1,0,44,61,
        1,0,0,0,45,46,3,10,5,0,46,47,5,7,0,0,47,48,3,4,2,0,48,49,6,2,-1,
        0,49,61,1,0,0,0,50,51,3,8,4,0,51,52,5,7,0,0,52,53,3,4,2,0,53,54,
        6,2,-1,0,54,61,1,0,0,0,55,56,3,12,6,0,56,57,5,7,0,0,57,58,3,4,2,
        0,58,59,6,2,-1,0,59,61,1,0,0,0,60,39,1,0,0,0,60,40,1,0,0,0,60,45,
        1,0,0,0,60,50,1,0,0,0,60,55,1,0,0,0,61,5,1,0,0,0,62,63,5,2,0,0,63,
        64,5,6,0,0,64,65,3,14,7,0,65,66,6,3,-1,0,66,7,1,0,0,0,67,68,5,4,
        0,0,68,69,5,6,0,0,69,70,5,2,0,0,70,71,6,4,-1,0,71,9,1,0,0,0,72,73,
        5,1,0,0,73,74,5,6,0,0,74,75,5,2,0,0,75,76,5,2,0,0,76,77,6,5,-1,0,
        77,11,1,0,0,0,78,79,5,4,0,0,79,80,5,6,0,0,80,81,5,1,0,0,81,82,6,
        6,-1,0,82,13,1,0,0,0,83,84,5,3,0,0,84,90,6,7,-1,0,85,86,5,3,0,0,
        86,87,3,14,7,0,87,88,6,7,-1,0,88,90,1,0,0,0,89,83,1,0,0,0,89,85,
        1,0,0,0,90,15,1,0,0,0,3,37,60,89
    ]

class metagrammarParser ( Parser ):

    grammarFileName = "metagrammar.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>",
                     "<INVALID>", "'ROOT'", "'->'", "';'" ]

    symbolicNames = [ "<INVALID>", "REL", "GROUP", "ENT", "COLL", "ROOT",
                      "PROD_SYMBOL", "PROD_SEPARATOR", "WS" ]

    RULE_start = 0
    RULE_rootList = 1
    RULE_ruleList = 2
    RULE_group = 3
    RULE_groupColl = 4
    RULE_relation = 5
    RULE_relationColl = 6
    RULE_entList = 7

    ruleNames =  [ "start", "rootList", "ruleList", "group", "groupColl",
                   "relation", "relationColl", "entList" ]

    EOF = Token.EOF
    REL=1
    GROUP=2
    ENT=3
    COLL=4
    ROOT=5
    PROD_SYMBOL=6
    PROD_SEPARATOR=7
    WS=8

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.2")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class StartContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self._rootList = None # RootListContext
            self._ruleList = None # RuleListContext

        def ROOT(self):
            return self.getToken(metagrammarParser.ROOT, 0)

        def PROD_SYMBOL(self):
            return self.getToken(metagrammarParser.PROD_SYMBOL, 0)

        def rootList(self):
            return self.getTypedRuleContext(metagrammarParser.RootListContext,0)


        def PROD_SEPARATOR(self):
            return self.getToken(metagrammarParser.PROD_SEPARATOR, 0)

        def ruleList(self):
            return self.getTypedRuleContext(metagrammarParser.RuleListContext,0)


        def EOF(self):
            return self.getToken(metagrammarParser.EOF, 0)

        def getRuleIndex(self):
            return metagrammarParser.RULE_start

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterStart" ):
                listener.enterStart(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitStart" ):
                listener.exitStart(self)




    def start(self):

        localctx = metagrammarParser.StartContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_start)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 16
            self.match(metagrammarParser.ROOT)
            self.state = 17
            self.match(metagrammarParser.PROD_SYMBOL)
            self.state = 18
            localctx._rootList = self.rootList()
            self.state = 19
            self.match(metagrammarParser.PROD_SEPARATOR)
            self.state = 20
            localctx._ruleList = self.ruleList()
            self.state = 21
            self.match(metagrammarParser.EOF)

            if not localctx._rootList.eL.issubset(localctx._ruleList.eL): raise ValueError("Some entities have not been defined : '" + ','.join(localctx._ruleList.eL))
            if not localctx._rootList.gL.issubset(localctx._ruleList.gL): raise ValueError("Some groups have not been defined : '" + ','.join(localctx._ruleList.gL))
            if not localctx._rootList.rL.issubset(localctx._ruleList.rL): raise ValueError("Some relations have not been defined : '" + ','.join(localctx._ruleList.rL))
            if not localctx._rootList.cL.issubset(localctx._ruleList.cL): raise ValueError("Some collections have not been defined : '" + ','.join(localctx._ruleList.cL))

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class RootListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.eL = None
            self.gL = None
            self.rL = None
            self.cL = None
            self.g = None # Token
            self._rootList = None # RootListContext
            self.r = None # Token
            self.c = None # Token

        def rootList(self):
            return self.getTypedRuleContext(metagrammarParser.RootListContext,0)


        def GROUP(self):
            return self.getToken(metagrammarParser.GROUP, 0)

        def REL(self):
            return self.getToken(metagrammarParser.REL, 0)

        def COLL(self):
            return self.getToken(metagrammarParser.COLL, 0)

        def getRuleIndex(self):
            return metagrammarParser.RULE_rootList

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterRootList" ):
                listener.enterRootList(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitRootList" ):
                listener.exitRootList(self)




    def rootList(self):

        localctx = metagrammarParser.RootListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_rootList)
        try:
            self.state = 37
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [7]:
                self.enterOuterAlt(localctx, 1)

                localctx.eL = set()
                localctx.gL = set()
                localctx.rL = set()
                localctx.cL = set()

                pass
            elif token in [2]:
                self.enterOuterAlt(localctx, 2)
                self.state = 25
                localctx.g = self.match(metagrammarParser.GROUP)
                self.state = 26
                localctx._rootList = self.rootList()

                if (None if localctx.g is None else localctx.g.text) in localctx._rootList.gL: raise ValueError("Group '" + (None if localctx.g is None else localctx.g.text) + "' already present in root")
                localctx.eL = localctx._rootList.eL
                localctx.gL = localctx._rootList.gL | {(None if localctx.g is None else localctx.g.text)}
                localctx.rL = localctx._rootList.rL
                localctx.cL = localctx._rootList.cL

                pass
            elif token in [1]:
                self.enterOuterAlt(localctx, 3)
                self.state = 29
                localctx.r = self.match(metagrammarParser.REL)
                self.state = 30
                localctx._rootList = self.rootList()

                if (None if localctx.r is None else localctx.r.text) in localctx._rootList.rL: raise ValueError("Relation '" + (None if localctx.r is None else localctx.r.text) + "' already present in root")
                localctx.eL = localctx._rootList.eL
                localctx.gL = localctx._rootList.gL
                localctx.rL = localctx._rootList.rL | {(None if localctx.r is None else localctx.r.text)}
                localctx.cL = localctx._rootList.cL

                pass
            elif token in [4]:
                self.enterOuterAlt(localctx, 4)
                self.state = 33
                localctx.c = self.match(metagrammarParser.COLL)
                self.state = 34
                localctx._rootList = self.rootList()

                if (None if localctx.c is None else localctx.c.text) in localctx._rootList.cL: raise ValueError("Collection '" + (None if localctx.c is None else localctx.c.text) + "' already present in root")
                localctx.eL = localctx._rootList.eL
                localctx.gL = localctx._rootList.gL
                localctx.rL = localctx._rootList.rL
                localctx.cL = localctx._rootList.cL | {(None if localctx.c is None else localctx.c.text)}

                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class RuleListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.eL = None
            self.gL = None
            self.rL = None
            self.cL = None
            self._group = None # GroupContext
            self._ruleList = None # RuleListContext
            self._relation = None # RelationContext
            self._groupColl = None # GroupCollContext
            self._relationColl = None # RelationCollContext

        def group(self):
            return self.getTypedRuleContext(metagrammarParser.GroupContext,0)


        def PROD_SEPARATOR(self):
            return self.getToken(metagrammarParser.PROD_SEPARATOR, 0)

        def ruleList(self):
            return self.getTypedRuleContext(metagrammarParser.RuleListContext,0)


        def relation(self):
            return self.getTypedRuleContext(metagrammarParser.RelationContext,0)


        def groupColl(self):
            return self.getTypedRuleContext(metagrammarParser.GroupCollContext,0)


        def relationColl(self):
            return self.getTypedRuleContext(metagrammarParser.RelationCollContext,0)


        def getRuleIndex(self):
            return metagrammarParser.RULE_ruleList

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterRuleList" ):
                listener.enterRuleList(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitRuleList" ):
                listener.exitRuleList(self)




    def ruleList(self):

        localctx = metagrammarParser.RuleListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_ruleList)
        try:
            self.state = 60
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,1,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)

                localctx.eL = set()
                localctx.gL = set()
                localctx.rL = set()
                localctx.cL = set()

                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 40
                localctx._group = self.group()
                self.state = 41
                self.match(metagrammarParser.PROD_SEPARATOR)
                self.state = 42
                localctx._ruleList = self.ruleList()

                if localctx._group.name in localctx._ruleList.gL: raise ValueError("Group '" + localctx._group.name + "' already defined")
                #if not localctx._group.eL.issubset(localctx._ruleList.eL): raise ValueError("Group reference undefined entities: " + localctx._group.eL)
                localctx.eL = localctx._ruleList.eL
                localctx.gL = localctx._ruleList.gL | {localctx._group.name}
                localctx.rL = localctx._ruleList.rL
                localctx.cL = localctx._ruleList.cL

                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 45
                localctx._relation = self.relation()
                self.state = 46
                self.match(metagrammarParser.PROD_SEPARATOR)
                self.state = 47
                localctx._ruleList = self.ruleList()

                if localctx._relation.name in localctx._ruleList.rL: raise ValueError("Relation '" + localctx._relation.name + "' already defined")
                if not localctx._relation.gL.issubset(localctx._ruleList.gL): raise ValueError("Relation reference undefined groups: " + str(localctx._relation.gL))
                localctx.eL = localctx._ruleList.eL
                localctx.gL = localctx._ruleList.gL
                localctx.rL = localctx._ruleList.rL | {localctx._relation.name}
                localctx.cL = localctx._ruleList.cL

                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 50
                localctx._groupColl = self.groupColl()
                self.state = 51
                self.match(metagrammarParser.PROD_SEPARATOR)
                self.state = 52
                localctx._ruleList = self.ruleList()

                if localctx._groupColl.name in localctx._ruleList.cL: raise ValueError("Group collection '" + localctx._groupColl.name + "' already defined")
                if localctx._groupColl.grpName not in localctx._ruleList.gL: raise ValueError("Collection of undefined groups: " + localctx._groupColl.grpName)
                localctx.eL = localctx._ruleList.eL
                localctx.gL = localctx._ruleList.gL
                localctx.rL = localctx._ruleList.rL
                localctx.cL = localctx._ruleList.cL | {localctx._groupColl.name}

                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 55
                localctx._relationColl = self.relationColl()
                self.state = 56
                self.match(metagrammarParser.PROD_SEPARATOR)
                self.state = 57
                localctx._ruleList = self.ruleList()

                if localctx._relationColl.name in localctx._ruleList.cL: raise ValueError("Relation collection '" + localctx._relationColl.name + "' already defined")
                if localctx._relationColl.relName not in localctx._ruleList.rL: raise ValueError("Collection of undefined relation: " + localctx._relationColl.relName)
                localctx.eL = localctx._ruleList.eL
                localctx.gL = localctx._ruleList.gL
                localctx.rL = localctx._ruleList.rL
                localctx.cL = localctx._ruleList.cL | {localctx._relationColl.name}

                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class GroupContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.name = None
            self.eL = None
            self.g = None # Token
            self._entList = None # EntListContext

        def PROD_SYMBOL(self):
            return self.getToken(metagrammarParser.PROD_SYMBOL, 0)

        def entList(self):
            return self.getTypedRuleContext(metagrammarParser.EntListContext,0)


        def GROUP(self):
            return self.getToken(metagrammarParser.GROUP, 0)

        def getRuleIndex(self):
            return metagrammarParser.RULE_group

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterGroup" ):
                listener.enterGroup(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitGroup" ):
                listener.exitGroup(self)




    def group(self):

        localctx = metagrammarParser.GroupContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_group)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 62
            localctx.g = self.match(metagrammarParser.GROUP)
            self.state = 63
            self.match(metagrammarParser.PROD_SYMBOL)
            self.state = 64
            localctx._entList = self.entList()

            localctx.name = (None if localctx.g is None else localctx.g.text)
            localctx.eL = localctx._entList.eL

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class GroupCollContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.name = None
            self.grpName = None
            self.c = None # Token
            self.g = None # Token

        def PROD_SYMBOL(self):
            return self.getToken(metagrammarParser.PROD_SYMBOL, 0)

        def COLL(self):
            return self.getToken(metagrammarParser.COLL, 0)

        def GROUP(self):
            return self.getToken(metagrammarParser.GROUP, 0)

        def getRuleIndex(self):
            return metagrammarParser.RULE_groupColl

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterGroupColl" ):
                listener.enterGroupColl(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitGroupColl" ):
                listener.exitGroupColl(self)




    def groupColl(self):

        localctx = metagrammarParser.GroupCollContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_groupColl)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 67
            localctx.c = self.match(metagrammarParser.COLL)
            self.state = 68
            self.match(metagrammarParser.PROD_SYMBOL)
            self.state = 69
            localctx.g = self.match(metagrammarParser.GROUP)

            localctx.name = (None if localctx.c is None else localctx.c.text)
            localctx.grpName = (None if localctx.g is None else localctx.g.text)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class RelationContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.name = None
            self.gL = None
            self.r = None # Token
            self.g1 = None # Token
            self.g2 = None # Token

        def PROD_SYMBOL(self):
            return self.getToken(metagrammarParser.PROD_SYMBOL, 0)

        def REL(self):
            return self.getToken(metagrammarParser.REL, 0)

        def GROUP(self, i:int=None):
            if i is None:
                return self.getTokens(metagrammarParser.GROUP)
            else:
                return self.getToken(metagrammarParser.GROUP, i)

        def getRuleIndex(self):
            return metagrammarParser.RULE_relation

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterRelation" ):
                listener.enterRelation(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitRelation" ):
                listener.exitRelation(self)




    def relation(self):

        localctx = metagrammarParser.RelationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_relation)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 72
            localctx.r = self.match(metagrammarParser.REL)
            self.state = 73
            self.match(metagrammarParser.PROD_SYMBOL)
            self.state = 74
            localctx.g1 = self.match(metagrammarParser.GROUP)
            self.state = 75
            localctx.g2 = self.match(metagrammarParser.GROUP)

            if (None if localctx.g1 is None else localctx.g1.text) == (None if localctx.g2 is None else localctx.g2.text): raise ValueError("Relation between equivalent groups: " + (None if localctx.g1 is None else localctx.g1.text))
            localctx.name = (None if localctx.r is None else localctx.r.text)
            localctx.gL = {(None if localctx.g1 is None else localctx.g1.text), (None if localctx.g2 is None else localctx.g2.text)}

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class RelationCollContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.name = None
            self.relName = None
            self.c = None # Token
            self.r = None # Token

        def PROD_SYMBOL(self):
            return self.getToken(metagrammarParser.PROD_SYMBOL, 0)

        def COLL(self):
            return self.getToken(metagrammarParser.COLL, 0)

        def REL(self):
            return self.getToken(metagrammarParser.REL, 0)

        def getRuleIndex(self):
            return metagrammarParser.RULE_relationColl

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterRelationColl" ):
                listener.enterRelationColl(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitRelationColl" ):
                listener.exitRelationColl(self)




    def relationColl(self):

        localctx = metagrammarParser.RelationCollContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_relationColl)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 78
            localctx.c = self.match(metagrammarParser.COLL)
            self.state = 79
            self.match(metagrammarParser.PROD_SYMBOL)
            self.state = 80
            localctx.r = self.match(metagrammarParser.REL)

            localctx.name = (None if localctx.c is None else localctx.c.text)
            localctx.relName = (None if localctx.r is None else localctx.r.text)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class EntListContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.eL = None
            self.e = None # Token
            self._entList = None # EntListContext

        def ENT(self):
            return self.getToken(metagrammarParser.ENT, 0)

        def entList(self):
            return self.getTypedRuleContext(metagrammarParser.EntListContext,0)


        def getRuleIndex(self):
            return metagrammarParser.RULE_entList

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterEntList" ):
                listener.enterEntList(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitEntList" ):
                listener.exitEntList(self)




    def entList(self):

        localctx = metagrammarParser.EntListContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_entList)
        try:
            self.state = 89
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,2,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 83
                localctx.e = self.match(metagrammarParser.ENT)
                localctx.eL = {(None if localctx.e is None else localctx.e.text)}
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 85
                localctx.e = self.match(metagrammarParser.ENT)
                self.state = 86
                localctx._entList = self.entList()

                if (None if localctx.e is None else localctx.e.text) in localctx._entList.eL: raise ValueError("Duplicate entity: " + (None if localctx.e is None else localctx.e.text))
                localctx.eL = localctx._entList.eL | {(None if localctx.e is None else localctx.e.text)}

                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx
