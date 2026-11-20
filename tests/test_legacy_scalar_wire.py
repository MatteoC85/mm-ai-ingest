"""Canonical scalar spelling is exact across the optimized and legacy oracle."""
from pathlib import Path
import random
import sys
import unittest
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from machinemind.evidence import legacy_compatibility as lc


def original_wire(node):
    if node.kind == 'map':
        return '["map",[' + ','.join('[' + lc._json(k) + ',' + original_wire(v) + ']'
            for k,v in node.value) + ']]'
    if node.kind in ('list','tuple'):
        children = (node.value.wire_children() if type(node.value) is lc._PackedFloats
            else '[' + ','.join(original_wire(v) for v in node.value) + ']')
        return '[' + lc._json(node.kind) + ',' + children + ']'
    return lc._json([node.kind,node.value])


class ScalarWireTests(unittest.TestCase):
    def test_exact_scalar_spelling_and_original_invalid_value_failure(self):
        randomizer = random.Random(59)
        values = [None,True,False,0,1,-1,2**300,-2**300,'','quote"\\\n\t\u0000',
            'è😀','\ud800',0.0,-0.0,5e-324,1.7976931348623157e308]
        values += [randomizer.uniform(-1e250,1e250) for _ in range(5000)]
        values += [float('inf'),float('-inf'),float('nan')]
        def result(function,node):
            try: return ('value',function(node))
            except Exception as exc: return ('error',type(exc),str(exc))
        for value in values:
            kind = {type(None):'null',bool:'bool',int:'int',str:'str',float:'float'}[type(value)]
            node = lc._Frozen(kind,value)
            self.assertEqual(result(lc._wire_json,node),result(original_wire,node))
        for kind,value in [('str',9),('int',True),('bool',1),('float',1),('null',0)]:
            node = lc._Frozen(kind,value)
            self.assertEqual(result(lc._wire_json,node),result(original_wire,node))

    def test_nested_shapes_and_packed_vectors_preserve_wire_bytes_and_digest(self):
        record = {'unicode':'è😀','text':'Observed operational condition. '*100,
            'nested':[{'n':1.25,'flag':True,'empty':None,'signed':-0.0} for _ in range(25)],
            'tuple':(1,'a',False),'vector':[float(i)/1000 for i in range(1536)]}
        limits = lc.LegacyLimits(32,8_000_000,128_000_000)
        node = lc._Freezer(limits).freeze(record)
        self.assertEqual(lc._wire_json(node),original_wire(node))
        self.assertEqual(lc._thaw(node),record)


if __name__ == '__main__': unittest.main()
