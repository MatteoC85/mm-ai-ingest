"""Offline exact journal comparisons and pure conversion reuse.

No network/database/provider calls. Benchmark uses the real bounded refinement
with synthetic records and the same request-local immutable numeric pool.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_lineage_copy_reuse import fixture
from machinemind.evidence import ask_input, legacy_compatibility as legacy
from machinemind.retrieval import receipt_producers as production


def outer_fixture(count=2, vector_size=16):
    base, rows = fixture(count, vector_size)
    trace = production._OuterRetrievalTrace.__new__(production._OuterRetrievalTrace)
    trace.__dict__.update(base.__dict__)
    trace.pending, trace.completed, trace.bonus_done = {}, None, False
    trace._live_outer_records = {}
    trace.invoke = lambda fn, req, *a, **kw: fn(*a, **kw)
    trace.request = object()
    return trace, rows


def copying_operator(rows, *, lineage):
    out = [dict(row) for row in rows]
    lineage(tuple(((0, index),) for index in range(len(rows))))
    return out


class EvidenceJournalCpuTests(unittest.TestCase):
    def test_mutable_original_and_copy_are_both_rechecked(self):
        for boundary in ('original', 'copy'):
            for field, value in (('embedding', [0.0, 1.0]), ('page_from', 9),
                                 ('chunk_full', 'different evidence'), ('similarity', True)):
                with self.subTest(boundary=boundary, field=field):
                    trace, rows = outer_fixture()
                    source = rows[0]
                    copied = deepcopy(source)
                    (source if boundary == 'original' else copied)[field] = value
                    with self.assertRaises(production.EvidenceProductionError):
                        trace.event('copy', source, copied)

    def test_copy_retains_exact_origin_and_private_witness_and_limit(self):
        trace, rows = outer_fixture()
        original = trace.entry(rows[0])
        copied = trace.event('copy', rows[0], dict(rows[0]))
        entry = trace.entry(copied)
        self.assertIs(entry[1], original[1])
        self.assertEqual(entry[2], original[2])
        self.assertIsNot(copied, rows[0])
        trace.max_bytes = trace.bytes
        with self.assertRaisesRegex(production.EvidenceProductionError, 'byte limit'):
            trace.event('copy', rows[0], dict(rows[0]))

    def test_callback_mutation_of_selected_or_unselected_original_is_rejected(self):
        for selected in (False, True):
            trace, rows = outer_fixture()
            def callback(values, *, lineage):
                values[0 if selected else 1]['embedding'][0] = -7.0
                result = [dict(values[0])]
                lineage((((0, 0),),))
                return result
            with self.assertRaises(production.EvidenceProductionError):
                trace.traced(callback, [rows], rows)

    def test_exact_locator_cache_still_checks_mutable_records(self):
        trace, rows = outer_fixture()
        with patch.object(production, 'adapt_candidate', wraps=production.adapt_candidate) as adapt:
            result = trace.operator(copying_operator, [rows], rows, same_locator=True)
        # New derived output conversion remains required. Existing input locator
        # conversion alone is reused; this does not admit or authorize a source.
        self.assertEqual(adapt.call_count, len(result))
        self.assertEqual([trace.entry(r)[2] for r in result], [trace.entry(r)[2] for r in rows])
        rows[0]['embedding'][0] = -9.0
        with self.assertRaises(production.EvidenceProductionError):
            trace.operator(copying_operator, [rows], rows, same_locator=True)

    def test_cached_conversion_cannot_merge_distinct_locators(self):
        trace, rows = outer_fixture()
        def merge(values, *, lineage):
            lineage((((0, 0), (0, 1)),))
            return [dict(values[0])]
        with self.assertRaisesRegex(production.EvidenceProductionError, 'different locators'):
            trace.operator(merge, [rows], rows, same_locator=True)

    def test_context_or_adapter_limit_change_cannot_reuse_conversion(self):
        for altered in ('context', 'limits'):
            trace, rows = outer_fixture(1)
            entry = trace.entry(rows[0]); old = trace._adapted_views[id(entry[1])]
            context = replace(old[1], provenance=replace(old[1].provenance, original_reference='different'))
            limits = replace(old[2], max_text_chars=old[2].max_text_chars + 1)
            trace._adapted_views[id(entry[1])] = (old[0], context if altered == 'context' else old[1],
                limits if altered == 'limits' else old[2], *old[3:])
            with patch.object(production, 'adapt_candidate', wraps=production.adapt_candidate) as adapt:
                trace.operator(copying_operator, [rows], rows, same_locator=True)
            self.assertGreaterEqual(adapt.call_count, 2)

    def test_readonly_sequence_reuses_immutable_witnesses_but_detects_mutation(self):
        trace, rows = outer_fixture()
        original = legacy._LegacyWitness.capture(rows, trace.limits.legacy)
        with patch.object(legacy._LegacyWitness, 'capture', wraps=legacy._LegacyWitness.capture) as capture:
            composed = trace.snapshot(rows)
        self.assertEqual(capture.call_count, 0)
        self.assertEqual(composed._tree, original._tree)
        self.assertEqual(composed.size_bytes, original.size_bytes)
        self.assertEqual(composed._limits_profile, original._limits_profile)
        self.assertEqual(composed.retained_metadata_bytes, original.retained_metadata_bytes)
        def mutation(values): values[0]['embedding'][0] = -1.0
        with self.assertRaises(production.EvidenceProductionError):
            trace.readonly(mutation)(rows)

    def test_composed_snapshot_preserves_allocation_limits_and_alias_occurrences(self):
        trace, rows = outer_fixture()
        repeated = [rows[0], rows[0]]
        original = legacy._LegacyWitness.capture(repeated, trace.limits.legacy)
        composed = trace.snapshot(repeated)
        self.assertEqual(original._limits_profile, composed._limits_profile)
        self.assertEqual(original.size_bytes, composed.size_bytes)
        for attribute, exact in zip(('max_nodes', 'max_bytes', 'max_depth'), original._limits_profile):
            low = replace(trace.limits.legacy, **{attribute: exact - 1})
            high = replace(trace.limits.legacy, **{attribute: exact})
            witnesses = tuple(trace.entry(row)[1] for row in repeated)
            with self.assertRaises(legacy.EvidenceAssemblyError):
                legacy._LegacyWitness.sequence('list', witnesses, low)
            accepted = legacy._LegacyWitness.sequence('list', witnesses, high)
            self.assertEqual(accepted._tree, original._tree)
        trace.max_bytes = original.size_bytes - 1
        with self.assertRaises(production.EvidenceProductionError): trace.snapshot(repeated)

    def test_untracked_or_reordered_sequence_is_never_treated_as_original(self):
        trace, rows = outer_fixture()
        untracked = [deepcopy(rows[0])]
        with patch.object(legacy._LegacyWitness, 'capture', wraps=legacy._LegacyWitness.capture) as capture:
            trace.snapshot(untracked)
        self.assertEqual(capture.call_count, 1)
        def reorder(values): values.reverse()
        with self.assertRaises(production.EvidenceProductionError): trace.readonly(reorder)(rows)

    def test_exact_float_comparisons_keep_types_signed_zero_and_nested_order(self):
        original = {'a': [float(i) for i in range(16)], 'b': {'x': (1.0, -0.0)}}
        witness = legacy._LegacyWitness.capture(original, legacy.LegacyLimits(12, 1000, 100000))
        self.assertTrue(witness.matches(deepcopy(original)))
        for mutate in (lambda x: x['a'].__setitem__(0, -0.0),
                       lambda x: x['a'].__setitem__(1, 1),
                       lambda x: x['a'].__setitem__(1, True),
                       lambda x: x['a'].__setitem__(2, float('nan')),
                       lambda x: x.update(b={'x': (1.0, 0.0)})):
            changed = deepcopy(original); mutate(changed)
            self.assertFalse(witness.matches(changed))
            self.assertFalse(ask_input._same_value(original, changed))
        reordered = {'b': original['b'], 'a': original['a']}
        self.assertFalse(witness.matches(reordered))
        self.assertFalse(ask_input._same_value(original, reordered))


def benchmark(output=None, rounds=5):
    import ast, cProfile, hashlib, json, pstats, statistics, subprocess, time
    from dataclasses import fields
    from test_ask_bounded_acquisition import AskBoundedAcquisitionTests as T
    from machinemind.retrieval import candidate_ranking as ranking, evidence_orchestration as orchestration
    repo = Path(__file__).resolve().parents[1]
    revision = 'c2815ae310a44251e0865787c35075d5380f7aea'
    targets = [(ask_input, None, '_same_value', 'machinemind/evidence/ask_input.py'),
        (legacy, None, '_matches_frozen', 'machinemind/evidence/legacy_compatibility.py'),
        (legacy, legacy._Freezer, 'freeze', 'machinemind/evidence/legacy_compatibility.py'),
        (legacy, legacy._LegacyWitness, 'capture', 'machinemind/evidence/legacy_compatibility.py'),
        (production, production._OuterRetrievalTrace, 'snapshot', 'machinemind/retrieval/receipt_producers.py'),
        (production, production._PreparationTrace, 'copies', 'machinemind/retrieval/receipt_producers.py'),
        (production, production._PreparationTrace, 'traced', 'machinemind/retrieval/receipt_producers.py'),
        (production, production._OuterRetrievalTrace, 'operator', 'machinemind/retrieval/receipt_producers.py'),
        (production, production._OuterRetrievalTrace, 'event', 'machinemind/retrieval/receipt_producers.py')]
    saved, baseline, sources = [], [], {}
    for module, cls, name, file in targets:
        owner = cls or module
        saved.append((owner, name, owner.__dict__[name] if isinstance(owner.__dict__[name], classmethod) else getattr(owner, name)))
        if file not in sources:
            sources[file] = subprocess.check_output(['git', 'show', revision + ':' + file], cwd=repo, text=True)
        tree = ast.parse(sources[file]); nodes = tree.body
        if cls:
            nodes = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls.__name__).body
        node = next(node for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
        was_classmethod = any(isinstance(d, ast.Name) and d.id == 'classmethod' for d in node.decorator_list)
        node.decorator_list = []
        env = dict(module.__dict__)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<R7-baseline>', 'exec'), env)
        # Functions must share the live module's globals so both variants call
        # exactly their matching current helpers, types and immutable pool.
        import types
        baseline.append((owner, name, types.FunctionType(env[name].__code__, module.__dict__, name,
                                                        env[name].__defaults__, env[name].__closure__)))
        baseline[-1][2].__kwdefaults__ = env[name].__kwdefaults__
        if was_classmethod:
            baseline[-1] = (owner, name, classmethod(baseline[-1][2]))
    def install(functions):
        for owner, name, fn in functions: setattr(owner, name, fn)
        production._legacy_same_value = ask_input._same_value
    results = {'baseline': [], 'candidate': []}; cpu_times = {'baseline': [], 'candidate': []}; profiles = {}; fingerprints = set(); accounting = set()
    T.setUpClass()
    try:
        for index in range(rounds):
            for label, functions in ((('baseline', baseline), ('candidate', saved)) if index % 2 == 0 else (('candidate', saved), ('baseline', baseline))):
                install(functions)
                pool = legacy._PayloadPool(128_000_000)
                with legacy._payload_scope(pool):
                    request, decision, _ = T().fixture()
                    trace, rows = outer_fixture(count=48, vector_size=1536)
                    trace.request = request
                    sr = ranking.V13ScoreCandidatesRuntime(**{f.name: getattr(T.m, f.name)
                                                              for f in fields(ranking.V13ScoreCandidatesRuntime)})
                    def score(q, values, **kw): return ranking.v13_score_candidates(q, values, runtime=sr, **kw)
                    facets = ranking.AssistantCoreMergeFacetCandidatesRuntime(
                        _assistant_core_candidate_stable_key=trace.readonly(T.m._assistant_core_candidate_stable_key),
                        _dedup_text_values=trace.readonly(T.m._dedup_text_values),
                        _v13_merge_candidates=lambda groups: trace.operator(ranking.v13_merge_candidates, groups, groups, same_locator=True))
                    rt = T.m._assistant_core_refine_retrieval_runtime()
                    names = ('_assistant_core_retrieval_query', '_assistant_core_scope_value', '_dedup_text_values',
                        '_v13_fallback_plan', '_assistant_core_facet_candidate_confidence', '_assistant_core_candidate_source_type', '_v13_evidence_metrics')
                    local = replace(rt, **{n: trace.readonly(getattr(rt, n)) for n in names},
                        _v13_current_budget=lambda: None, _v13_initial_retrieval=lambda **kw: {'candidates': rows},
                        _v13_score_candidates=lambda q, values: trace.operator(score, [values], q, values),
                        _assistant_core_merge_facet_candidates=lambda groups: trace.run(ranking.assistant_core_merge_facet_candidates,
                            groups, runtime=facets, lineage=trace.event))
                    profile = cProfile.Profile() if index == rounds-1 else None
                    if profile: profile.enable()
                    started = time.perf_counter(); cpu_started = time.process_time()
                    result = orchestration.assistant_core_refine_retrieval(request, {'candidates': rows, 'citations': rows},
                        decision, runtime=local, lineage=trace.event)
                    elapsed = time.perf_counter()-started; cpu_elapsed = time.process_time()-cpu_started
                    if profile: profile.disable()
                    encoded = json.dumps(result, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
                    fingerprints.add(hashlib.sha256(encoded.encode()).hexdigest())
                    accounting.add((trace.bytes, trace._witness_budget.logical_bytes, len(trace.entries), len(trace.roots)))
                    if profile:
                        stats = pstats.Stats(profile)
                        profiles[label] = {'wall_seconds': elapsed, 'calls': stats.total_calls,
                            'top_cumulative': [{'file':Path(k[0]).name,'function':k[2],'calls':v[1], 'cumulative_seconds':round(v[3],6)}
                                for k,v in sorted(stats.stats.items(), key=lambda item:item[1][3], reverse=True)[:15]]}
                    else:
                        results[label].append(elapsed); cpu_times[label].append(cpu_elapsed)
                pool.close()
    finally:
        install(saved); T.tearDownClass()
    assert len(fingerprints) == 1 and len(accounting) == 1, 'Output or allocation changed'
    medians = {k: statistics.median(v) for k,v in results.items()}
    cpu_medians = {k: statistics.median(v) for k,v in cpu_times.items()}
    report = {'schema':'machinemind.offline-journal-performance.v1', 'baseline_commit':revision,
        'fixture':{'records':48,'facets':4,'vector_components':1536,'provider_calls':0,'database_calls':0},
        'wall_seconds':results,'median_seconds':medians,
        'cpu_seconds':cpu_times,'median_cpu_seconds':cpu_medians,
        'cpu_improvement_pct':100*(1-cpu_medians['candidate']/cpu_medians['baseline']),
        'improvement_pct':100*(1-medians['candidate']/medians['baseline']),
        'same_output_sha256':next(iter(fingerprints)),'same_accounting':list(next(iter(accounting))),
        'profiles':profiles,
        'code_sha256':{f:hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in sources},
        'limitation':'Synthetic offline refinement; no claim of live latency or semantic quality.'}
    if output:
        Path(output).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    if '--benchmark' in sys.argv:
        benchmark(sys.argv[sys.argv.index('--output')+1] if '--output' in sys.argv else None)
    else: unittest.main()
