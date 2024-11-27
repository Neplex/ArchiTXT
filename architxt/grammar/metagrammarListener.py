# Generated from metagrammar.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .metagrammarParser import metagrammarParser
else:
    from metagrammarParser import metagrammarParser

# This class defines a complete listener for a parse tree produced by metagrammarParser.
class metagrammarListener(ParseTreeListener):

    # Enter a parse tree produced by metagrammarParser#start.
    def enterStart(self, ctx:metagrammarParser.StartContext):
        pass

    # Exit a parse tree produced by metagrammarParser#start.
    def exitStart(self, ctx:metagrammarParser.StartContext):
        pass


    # Enter a parse tree produced by metagrammarParser#rootList.
    def enterRootList(self, ctx:metagrammarParser.RootListContext):
        pass

    # Exit a parse tree produced by metagrammarParser#rootList.
    def exitRootList(self, ctx:metagrammarParser.RootListContext):
        pass


    # Enter a parse tree produced by metagrammarParser#ruleList.
    def enterRuleList(self, ctx:metagrammarParser.RuleListContext):
        pass

    # Exit a parse tree produced by metagrammarParser#ruleList.
    def exitRuleList(self, ctx:metagrammarParser.RuleListContext):
        pass


    # Enter a parse tree produced by metagrammarParser#group.
    def enterGroup(self, ctx:metagrammarParser.GroupContext):
        pass

    # Exit a parse tree produced by metagrammarParser#group.
    def exitGroup(self, ctx:metagrammarParser.GroupContext):
        pass


    # Enter a parse tree produced by metagrammarParser#groupColl.
    def enterGroupColl(self, ctx:metagrammarParser.GroupCollContext):
        pass

    # Exit a parse tree produced by metagrammarParser#groupColl.
    def exitGroupColl(self, ctx:metagrammarParser.GroupCollContext):
        pass


    # Enter a parse tree produced by metagrammarParser#relation.
    def enterRelation(self, ctx:metagrammarParser.RelationContext):
        pass

    # Exit a parse tree produced by metagrammarParser#relation.
    def exitRelation(self, ctx:metagrammarParser.RelationContext):
        pass


    # Enter a parse tree produced by metagrammarParser#relationColl.
    def enterRelationColl(self, ctx:metagrammarParser.RelationCollContext):
        pass

    # Exit a parse tree produced by metagrammarParser#relationColl.
    def exitRelationColl(self, ctx:metagrammarParser.RelationCollContext):
        pass


    # Enter a parse tree produced by metagrammarParser#entList.
    def enterEntList(self, ctx:metagrammarParser.EntListContext):
        pass

    # Exit a parse tree produced by metagrammarParser#entList.
    def exitEntList(self, ctx:metagrammarParser.EntListContext):
        pass



del metagrammarParser
