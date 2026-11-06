"""Real bounded directory, fake GET responses; no network or provider calls."""
from pathlib import Path
import json
import sys
import threading
import time
import unittest
from collections import Counter
from urllib.parse import urlsplit, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machinemind.authority.bubble_directory import BubbleConnection, BubbleDirectory
from machinemind.authority.contracts import ApplicationBoundary, AuthorityError, AuthorityLimits, AuthorityMeter
from machinemind.authority.policy import BubbleAuthority, BubbleSchema, SourceSchema
from machinemind.evidence.contracts import SourceType
from machinemind.retrieval.chunk_evidence import ChunkReadScope


class Response:
    status = 200
    headers = {'Content-Type': 'application/json'}

    def __init__(self, value):
        self.data = json.dumps({'response': value}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, limit):
        return self.data[:limit]


class FakeGET:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = []
        self.hook = lambda kind, record, constraints: None
        self.machine_company = 'company-a'
        self.unfiltered_steps = False
        self.rows = {kind.value: [{'_id': kind.value + '-a', 'company': 'company-a',
            'machine': 'machine-a', 'deleted': False}] for kind in SourceType if kind != SourceType.STEP}
        self.rows['step'] = [{'_id': 'step-a', 'procedure': 'procedure-a', 'deleted': False}]

    def open(self, request, timeout):
        assert request.get_method() == 'GET'
        parsed = urlsplit(request.full_url)
        tail = parsed.path.split('/obj/')[1].split('/')
        kind, record = tail[0], tail[1] if len(tail) == 2 else None
        params = parse_qs(parsed.query)
        constraints = json.loads(params['constraints'][0]) if params else []
        with self.lock:
            self.calls.append((kind, record, constraints))
        self.hook(kind, record, constraints)
        if record:
            value = {'_id': record}
            if kind == 'machine':
                value['company'] = self.machine_company
        else:
            rows = list(self.rows[kind])
            if kind == 'step' and not self.unfiltered_steps:
                allowed = constraints[0]['value']
                rows = [r for r in rows if r.get('procedure') in allowed]
            cursor = int(params['cursor'][0])
            limit = int(params['limit'][0])
            page = rows[cursor:cursor + limit]
            value = {'results': page, 'count': len(page), 'remaining': len(rows) - cursor - len(page), 'cursor': cursor}
        return Response(value)


class AuthorityCatalogPipelineTests(unittest.TestCase):
    def fixture(self, *, parallel=True):
        fake = FakeGET()
        meter = AuthorityMeter(AuthorityLimits(80, 100000, 1000000, 1000, 100, 2, 60), time.monotonic)
        directory = BubbleDirectory(connection=BubbleConnection('https://offline.invalid/api/1.1/obj',
            'offline.invalid', 'offline-token'), meter=meter, opener=fake, request_owned_io=parallel)
        boundary = ApplicationBoundary(secret='a' * 32, legacy_secret='different-offline-secret')
        schema = BubbleSchema('company', 'machine', 'company', tuple(
            SourceSchema(kind, kind.value, None if kind == SourceType.STEP else 'company',
                None if kind == SourceType.STEP else 'machine', 'deleted',
                parent_field='procedure' if kind == SourceType.STEP else None) for kind in SourceType))
        authority = BubbleAuthority(directory=directory, schema=schema, boundary=boundary)
        grant = boundary.authenticate(supplied_secret='a' * 32)
        scope = ChunkReadScope('company-a', 'machine-a', 'machine_all')
        self.addCleanup(directory.close)
        return fake, directory, authority, grant, scope

    def invoke(self, authority, grant, scope, expected):
        return authority.current_sources(grant, scope) if expected is None else authority.validate_sources(grant, scope, expected)

    def test_step_io_overlaps_other_catalog_but_no_allowance_until_all_join(self):
        for observed in (False, True):
            with self.subTest(observed=observed):
                fake, directory, authority, grant, scope = self.fixture()
                expected = authority.current_sources(grant, scope) if observed else None
                fake.calls.clear()
                document_entered, step_entered, release_document, release_step, done = [threading.Event() for _ in range(5)]
                values, errors = [], []

                def hook(kind, record, constraints):
                    if kind == 'document':
                        document_entered.set()
                        if not release_document.wait(3):
                            raise AssertionError('document release missing')
                    if kind == 'step':
                        step_entered.set()
                        if not release_step.wait(3):
                            raise AssertionError('step release missing')

                fake.hook = hook
                def run():
                    try:
                        values.append(self.invoke(authority, grant, scope, expected))
                    except BaseException as exc:
                        errors.append(exc)
                    finally:
                        done.set()
                worker = threading.Thread(target=run)
                worker.start()
                try:
                    self.assertTrue(document_entered.wait(2))
                    self.assertTrue(step_entered.wait(2), 'Step must start before the slow independent catalog returns')
                    self.assertFalse(done.is_set())
                    self.assertFalse(values)
                    self.assertEqual(sum(record is not None for _, record, _ in fake.calls), 0 if observed else 2)
                    release_step.set()
                    self.assertFalse(done.is_set(), 'No allowance before the independent catalog joins')
                finally:
                    release_document.set()
                    release_step.set()
                    worker.join(4)
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(len(values[0]), 6)
                self.assertEqual(len(fake.calls), 8 if observed else 10)
                self.assertEqual(sum(record is not None for _, record, _ in fake.calls), 2 if observed else 4)
                self.assertFalse(directory.meter._tickets)

    def test_sequential_and_parallel_return_identical_identity_and_read_multiset(self):
        results = []
        for parallel in (False, True):
            fake, directory, authority, grant, scope = self.fixture(parallel=parallel)
            current = authority.current_sources(grant, scope)
            validated = authority.validate_sources(grant, scope, current)
            results.append((current, validated, Counter((k, r, json.dumps(c, sort_keys=True)) for k, r, c in fake.calls)))
            self.assertEqual(directory.meter.calls, 18)
        self.assertEqual(results[0], results[1])

    def test_step_batches_are_bounded_and_exact_without_extra_queries(self):
        fake, _, authority, grant, scope = self.fixture()
        fake.rows['procedure'] = [{'_id': 'p-' + str(i), 'company': 'company-a', 'machine': 'machine-a', 'deleted': False} for i in range(51)]
        fake.rows['step'] = [{'_id': 's-' + str(i), 'procedure': 'p-' + str(i), 'deleted': False} for i in range(51)]
        result = authority.current_sources(grant, scope)
        batches = [c[0]['value'] for k, _, c in fake.calls if k == 'step']
        self.assertEqual([len(b) for b in batches], [50, 1])
        self.assertEqual([uid for b in batches for uid in b], [r['_id'] for r in fake.rows['procedure']])
        self.assertEqual(len(result), 106)
        self.assertEqual(len(fake.calls), 11)

    def test_revoked_parent_excludes_children_and_observed_fence_rejects(self):
        fake, _, authority, grant, scope = self.fixture()
        expected = authority.current_sources(grant, scope)
        fake.rows['procedure'][0]['deleted'] = True
        fake.calls.clear()
        current = authority.current_sources(grant, scope)
        self.assertFalse(any(s.source_type in (SourceType.PROCEDURE, SourceType.STEP) for s in current))
        self.assertFalse(any(k == 'step' for k, _, _ in fake.calls))
        with self.assertRaisesRegex(AuthorityError, 'SCOPE_DENIED'):
            authority.validate_sources(grant, scope, expected)

    def test_invalid_parent_never_dispatches_step(self):
        for mutation in ({'company': 'other-company'}, {'deleted': 'false'}, {'machine': '__MM_COMPANY_GENERAL__'}):
            with self.subTest(mutation=mutation):
                fake, _, authority, grant, scope = self.fixture()
                fake.rows['procedure'][0].update(mutation)
                with self.assertRaises(AuthorityError):
                    authority.current_sources(grant, scope)
                self.assertFalse(any(k == 'step' for k, _, _ in fake.calls))

    def test_sibling_failure_joins_pending_step_and_returns_no_partial_grant(self):
        fake, directory, authority, grant, scope = self.fixture()
        step_entered, release_step, done = [threading.Event() for _ in range(3)]
        errors, values = [], []
        def hook(kind, record, constraints):
            if kind == 'step':
                step_entered.set()
                if not release_step.wait(3):
                    raise AssertionError('release missing')
            if kind == 'document':
                if not step_entered.wait(3):
                    raise AssertionError('step missing')
                raise AuthorityError('AUTHORITY_RESPONSE_INVALID')
        fake.hook = hook
        def run():
            try:
                values.append(authority.current_sources(grant, scope))
            except BaseException as exc:
                errors.append(exc)
            finally:
                done.set()
        worker = threading.Thread(target=run)
        worker.start()
        try:
            self.assertTrue(step_entered.wait(2))
            self.assertFalse(done.is_set())
        finally:
            release_step.set()
            worker.join(4)
        self.assertFalse(worker.is_alive())
        self.assertEqual(values, [])
        self.assertEqual([getattr(e, 'code', None) for e in errors], ['AUTHORITY_RESPONSE_INVALID'])
        self.assertFalse(directory.meter._tickets)
        self.assertEqual(sum(record is not None for _, record, _ in fake.calls), 2)

    def test_final_machine_reassignment_still_denies(self):
        fake, _, authority, grant, scope = self.fixture()
        def hook(kind, record, constraints):
            if kind == 'step':
                fake.machine_company = 'other-company'
        fake.hook = hook
        with self.assertRaisesRegex(AuthorityError, 'CONTEXT_MISMATCH'):
            authority.current_sources(grant, scope)
        self.assertEqual(sum(record is not None for _, record, _ in fake.calls), 4)

    def test_foreign_step_parent_is_rejected_after_join(self):
        fake, _, authority, grant, scope = self.fixture()
        fake.unfiltered_steps = True
        fake.rows['step'][0]['procedure'] = 'foreign-procedure'
        with self.assertRaisesRegex(AuthorityError, 'AUTHORITY_RESPONSE_INVALID'):
            authority.current_sources(grant, scope)
        self.assertEqual(sum(record is not None for _, record, _ in fake.calls), 2)

    def test_observed_document_only_does_not_read_procedure_or_step(self):
        fake, _, authority, grant, scope = self.fixture()
        expected = frozenset(s for s in authority.current_sources(grant, scope) if s.source_type == SourceType.DOCUMENT)
        fake.calls.clear()
        self.assertEqual(authority.validate_sources(grant, scope, expected), expected)
        self.assertEqual(Counter(k for k, _, _ in fake.calls), Counter(document=1, company=1, machine=1))


if __name__ == '__main__':
    unittest.main()
