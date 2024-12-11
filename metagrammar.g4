grammar metagrammar;

REL: 'REL::' [a-zA-Z0-9]+ ('<->' [a-zA-Z0-9]+)?;
GROUP: 'GROUP::' [a-zA-Z0-9]+;
ENT: 'ENT::' [a-zA-Z0-9]+;
COLL: 'COLL::' [a-zA-Z0-9]+ ('<->' [a-zA-Z0-9]+)?;
ROOT: 'ROOT';
PROD_SYMBOL: '->';
PROD_SEPARATOR: ';';
WS: [ \t\r\n]+ -> skip;

start
    : ROOT PROD_SYMBOL rootList PROD_SEPARATOR ruleList EOF {
if not $rootList.eL.issubset($ruleList.eL): raise ValueError("Some entities have not been defined : '" + ','.join($ruleList.eL))
if not $rootList.gL.issubset($ruleList.gL): raise ValueError("Some groups have not been defined : '" + ','.join($ruleList.gL))
if not $rootList.rL.issubset($ruleList.rL): raise ValueError("Some relations have not been defined : '" + ','.join($ruleList.rL))
if not $rootList.cL.issubset($ruleList.cL): raise ValueError("Some collections have not been defined : '" + ','.join($ruleList.cL))
}
    ;

rootList returns [set eL, set gL, set rL, set cL]
    : {
$eL = set()
$gL = set()
$rL = set()
$cL = set()
}
    | g=GROUP rootList {
if $g.text in $rootList.gL: raise ValueError("Group '" + $g.text + "' already present in root")
$eL = $rootList.eL
$gL = $rootList.gL | {$g.text}
$rL = $rootList.rL
$cL = $rootList.cL
}
    | r=REL rootList {
if $r.text in $rootList.rL: raise ValueError("Relation '" + $r.text + "' already present in root")
$eL = $rootList.eL
$gL = $rootList.gL
$rL = $rootList.rL | {$r.text}
$cL = $rootList.cL
}
    | c=COLL rootList {
if $c.text in $rootList.cL: raise ValueError("Collection '" + $c.text + "' already present in root")
$eL = $rootList.eL
$gL = $rootList.gL
$rL = $rootList.rL
$cL = $rootList.cL | {$c.text}
}
    ;

ruleList returns [set eL, set gL, set rL, set cL]
    : {
$eL = set()
$gL = set()
$rL = set()
$cL = set()
}
    | group PROD_SEPARATOR ruleList {
if $group.name in $ruleList.gL: raise ValueError("Group '" + $group.name + "' already defined")
#if not $group.eL.issubset($ruleList.eL): raise ValueError("Group reference undefined entities: " + $group.eL)
$eL = $ruleList.eL
$gL = $ruleList.gL | {$group.name}
$rL = $ruleList.rL
$cL = $ruleList.cL
}
    | relation PROD_SEPARATOR ruleList {
if $relation.name in $ruleList.rL: raise ValueError("Relation '" + $relation.name + "' already defined")
if not $relation.gL.issubset($ruleList.gL): raise ValueError("Relation reference undefined groups: " + str($relation.gL))
$eL = $ruleList.eL
$gL = $ruleList.gL
$rL = $ruleList.rL | {$relation.name}
$cL = $ruleList.cL
}
    | groupColl PROD_SEPARATOR ruleList {
if $groupColl.name in $ruleList.cL: raise ValueError("Group collection '" + $groupColl.name + "' already defined")
if $groupColl.grpName not in $ruleList.gL: raise ValueError("Collection of undefined groups: " + $groupColl.grpName)
$eL = $ruleList.eL
$gL = $ruleList.gL
$rL = $ruleList.rL
$cL = $ruleList.cL | {$groupColl.name}
}
    | relationColl PROD_SEPARATOR ruleList {
if $relationColl.name in $ruleList.cL: raise ValueError("Relation collection '" + $relationColl.name + "' already defined")
if $relationColl.relName not in $ruleList.rL: raise ValueError("Collection of undefined relation: " + $relationColl.relName)
$eL = $ruleList.eL
$gL = $ruleList.gL
$rL = $ruleList.rL
$cL = $ruleList.cL | {$relationColl.name}
}
    ;


group returns [str name, set eL]
    : g=GROUP PROD_SYMBOL entList {
$name = $g.text
$eL = $entList.eL
}
    ;

groupColl returns [str name, str grpName]
    : c=COLL PROD_SYMBOL g=GROUP {
$name = $c.text
$grpName = $g.text
}
    ;

relation returns [str name, set gL]
    : r=REL PROD_SYMBOL g1=GROUP g2=GROUP {
if $g1.text == $g2.text: raise ValueError("Relation between equivalent groups: " + $g1.text)
$name = $r.text
$gL = {$g1.text, $g2.text}
}
    ;

relationColl returns [str name, str relName]
    : c=COLL PROD_SYMBOL r=REL {
$name = $c.text
$relName = $r.text
}
    ;

entList returns [set eL]
    : e=ENT {$eL = {$e.text}}
    | e=ENT entList {
if $e.text in $entList.eL: raise ValueError("Duplicate entity: " + $e.text)
$eL = $entList.eL | {$e.text}
}
    ;
