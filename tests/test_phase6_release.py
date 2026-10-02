"""Offline release fault tests. No subprocess, provider or network is contacted."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import phase6_release as guard


COMMIT = '2' * 40
PRIOR_COMMIT = '1' * 40
BUILD = '12345678-1234-1234-1234-123456789012'
DIGEST = 'sha256:' + 'a' * 64
PRIOR = guard.SERVICE + '-prior'
CANDIDATE = guard.SERVICE + '-b' + BUILD.replace('-', '')


def ready_object(name, spec):
    return {'metadata': {'name': name, 'generation': 1}, 'spec': spec,
            'status': {'observedGeneration': 1,
                       'conditions': [{'type': 'Ready', 'status': 'True'}]}}


class FakeCloud:
    """Simulate Cloud Run state transitions, without bypassing guard validation."""
    def __init__(self, fault=None):
        self.fault = fault
        self.commands = []
        self.events = []
        self.staged = False
        self.promoted = False
        env = [
            {'name': 'COMMIT_SHA', 'value': PRIOR_COMMIT},
            {'name': 'MM_ASK_REQUEST_AUTHORITY', 'value': 'required'},
            {'name': 'MM_INGEST_LEDGER_AUTO_DDL', 'value': '0'},
        ]
        env += [{'name': name, 'valueFrom': {'secretKeyRef': reference}}
                for name, reference in guard.SECRET_REFS.items()]
        self.before_spec = {'containers': [{'image': guard.IMAGE_ROOT + '@' + DIGEST,
                                            'env': env}], 'timeoutSeconds': 300}
        self.candidate_spec = copy.deepcopy(self.before_spec)
        self.candidate_spec['containers'][0]['env'][0]['value'] = COMMIT
        self.candidate_spec['containers'][0]['env'] += [
            {'name': 'MM_USAGE_ENFORCEMENT', 'value': 'required'},
            {'name': 'MM_USAGE_TIMEZONE', 'value': 'Europe/Rome'},
            {'name': 'MM_USAGE_AUTHORITY_SECRET',
             'valueFrom': {'secretKeyRef': guard.USAGE_SECRET}},
        ]

    def service(self):
        active = CANDIDATE if self.promoted else PRIOR
        if self.staged and self.fault == 'traffic':
            active = CANDIDATE
        spec = copy.deepcopy(self.candidate_spec if self.staged else self.before_spec)
        if self.staged and self.fault == 'config':
            spec['timeoutSeconds'] = 301
        obj = ready_object(guard.SERVICE, {'template': {'spec': spec},
                                           'traffic': [{'revisionName': active, 'percent': 100}]})
        obj['status'].update(traffic=[{'revisionName': active, 'percent': 100}],
                             url='https://service.example.run.app')
        if self.staged:
            obj['status']['traffic'].append({'revisionName': CANDIDATE, 'percent': 0,
                'tag': 'p6-candidate', 'url': 'https://candidate.example.run.app'})
        return obj

    def gcloud(self, *args):
        self.commands.append(args)
        if args[:2] == ('run', 'deploy'):
            if self.fault == 'deploy':
                raise guard.GuardError('GCLOUD_COMMAND_FAILED')
            self.staged = True
            self.events.append('stage')
            return ''
        if args[:3] == ('run', 'services', 'update-traffic'):
            self.promoted = True
            self.events.append('promote')
            return ''
        if args[:3] == ('run', 'services', 'describe'):
            if self.staged and self.fault == 'service_read':
                raise guard.GuardError('GCLOUD_COMMAND_FAILED')
            return json.dumps(self.service())
        if args[:3] == ('run', 'revisions', 'describe'):
            name = args[3]
            if name == CANDIDATE and self.fault == 'candidate_read':
                raise guard.GuardError('GCLOUD_COMMAND_FAILED')
            spec = self.candidate_spec if name == CANDIDATE else self.before_spec
            obj = ready_object(name, copy.deepcopy(spec))
            obj['status']['imageDigest'] = guard.IMAGE_ROOT + '@' + DIGEST
            if name == CANDIDATE and self.fault == 'not_ready':
                obj['status']['conditions'][0]['status'] = 'False'
            return json.dumps(obj)
        if args[:3] == ('run', 'revisions', 'list'):
            names = [PRIOR] + ([CANDIDATE] if self.staged else [])
            if self.staged and self.fault == 'concurrent_revision':
                names.append(guard.SERVICE + '-concurrent')
            return json.dumps([{'metadata': {'name': name}} for name in names])
        if args[:4] == ('artifacts', 'docker', 'images', 'describe'):
            return DIGEST
        raise AssertionError('Unexpected command family in offline test')

    def health(self, url, commit, revision, **kwargs):
        self.events.append('health')
        if self.fault == 'health':
            raise guard.GuardError('USAGE_NATIVE_HEALTH_NOT_READY')
        assert commit == COMMIT and revision == CANDIDATE
        assert kwargs == {'usage_required': True}


class ReleaseTests(unittest.TestCase):
    def execute(self, cloud, *, operation='deploy'):
        with patch.object(guard, 'gcloud', side_effect=cloud.gcloud), \
             patch.object(guard, 'health', side_effect=cloud.health), \
             contextlib.redirect_stdout(io.StringIO()):
            return guard.execute(guard.PROJECT, COMMIT, BUILD, operation, PRIOR_COMMIT,
                                 usage_required=True)

    def test_stage_preserves_legacy_usage_health_without_promoting(self):
        cloud = FakeCloud()
        result = self.execute(cloud, operation='stage')
        self.assertEqual(result['status'], 'PASS_REQUIRED_STAGED')
        self.assertEqual(cloud.events, ['stage', 'health'])
        self.assertTrue(result['usage_native_health_verified'])
        self.assertFalse(cloud.promoted)
        self.assertFalse(any(args[:3] == ('run', 'services', 'update-traffic')
                             for args in cloud.commands))

    def test_required_deploy_checks_health_before_exact_promotion(self):
        cloud = FakeCloud()
        result = self.execute(cloud)
        self.assertEqual(result['status'], 'PASS_REQUIRED_DEPLOY')
        self.assertEqual(cloud.events, ['stage', 'health', 'promote', 'health'])
        deploy = next(args for args in cloud.commands if args[:2] == ('run', 'deploy'))
        self.assertIn('--no-traffic', deploy)
        self.assertIn('--image=' + guard.IMAGE_ROOT + '@' + DIGEST, deploy)
        self.assertIn('--update-secrets=MM_USAGE_AUTHORITY_SECRET=mm-usage-authority-v6:1', deploy)
        promotion = next(args for args in cloud.commands
                         if args[:3] == ('run', 'services', 'update-traffic'))
        self.assertIn('--to-revisions=' + CANDIDATE + '=100', promotion)

    def test_pre_promotion_failures_do_not_promote(self):
        failures = (
            ('deploy', 'GCLOUD_COMMAND_FAILED', 'stage_no_traffic'),
            ('service_read', 'GCLOUD_COMMAND_FAILED', 'read_staged_service'),
            ('candidate_read', 'GCLOUD_COMMAND_FAILED', 'read_candidate_revision'),
            ('traffic', 'TRAFFIC_CHANGED_BEFORE_VALIDATION', 'validate_staged_traffic'),
            ('concurrent_revision', 'CONCURRENT_OR_UNEXPECTED_REVISION', 'validate_staged_revision_set'),
            ('config', 'UNRELATED_CONFIG_CHANGED', 'validate_staged_configuration'),
            ('not_ready', 'RESOURCE_NOT_READY', 'validate_staged_configuration'),
            ('health', 'USAGE_NATIVE_HEALTH_NOT_READY', 'check_candidate_identity_before_promotion'),
        )
        for fault, code, phase in failures:
            with self.subTest(fault=fault):
                cloud = FakeCloud(fault)
                with self.assertRaisesRegex(guard.GuardError, '^' + code + '$'):
                    self.execute(cloud)
                self.assertEqual(guard.PHASE, phase)
                self.assertFalse(cloud.promoted)
                self.assertNotIn('promote', cloud.events)


class QualityCloud(FakeCloud):
    BASE = 'mm-ai-ingest-prod-bc1a194bea02e4191a33773d371238af2'
    SHA = '3608ee56b2eb292c35c34558e5ae998425a26504'
    BASE_DIGEST = 'sha256:6db9b12ca70c6e11f1a6fd2d7e69fbb1a1901b62b155cf320bff82ee434f4656'
    FAILED = 'mm-ai-ingest-prod-bf992ccb586fc4cf1830480a55145a9bb'
    FAILED_SHA = '201e9492fb5fc061f1702ec8925c051567963125'
    FAILED_DIGEST = 'sha256:2ad3ff7448219e818fe705fdba4c906830b45ca163f4956602bd27c74c7b3871'

    def __init__(self, fault=None, *, dirty=True):
        super().__init__(fault)
        self.dirty = dirty
        self.traffic_revision = self.BASE
        self.annotations = {'autoscaling.knative.dev/maxScale': '3',
                            'run.googleapis.com/cloudsql-instances': 'fixture:region:db'}
        self.before_spec['serviceAccountName'] = 'fixture-service-account'
        self.before_spec['containerConcurrency'] = 80
        self.before_spec['containers'][0]['resources'] = {'limits': {'cpu': '1', 'memory': '512Mi'}}
        self.before_spec['containers'][0]['image'] = guard.IMAGE_ROOT + '@' + self.BASE_DIGEST
        self.before_spec['containers'][0]['env'][0]['value'] = self.SHA
        self.failed_spec = copy.deepcopy(self.before_spec)
        self.failed_spec['containers'][0]['image'] = guard.IMAGE_ROOT + '@' + self.FAILED_DIGEST
        self.failed_spec['containers'][0]['env'][0]['value'] = self.FAILED_SHA
        self.failed_spec['containers'][0]['env'] += [
            {'name': 'MM_USAGE_ENFORCEMENT', 'value': 'required'},
            {'name': 'MM_USAGE_TIMEZONE', 'value': 'Europe/Rome'},
            {'name': 'MM_USAGE_AUTHORITY_SECRET', 'valueFrom': {'secretKeyRef': dict(guard.USAGE_SECRET)}},
        ]
        self.candidate_spec = None

    def active_revision(self):
        obj = ready_object(self.BASE, copy.deepcopy(self.before_spec))
        obj['metadata']['annotations'] = {**self.annotations, 'run.googleapis.com/ingress': 'all'}
        obj['status']['imageDigest'] = guard.IMAGE_ROOT + '@' + self.BASE_DIGEST
        return obj

    def service(self):
        active = self.traffic_revision
        spec = copy.deepcopy(self.candidate_spec if self.staged else
                             self.failed_spec if self.dirty else self.before_spec)
        obj = ready_object(guard.SERVICE, {'template': {'spec': spec, 'metadata': {'annotations': dict(self.annotations)}},
                'traffic': [{'revisionName': active, 'percent': 100}]})
        obj['metadata']['annotations'] = {'run.googleapis.com/ingress': 'all'}
        obj['status'].update(traffic=[{'revisionName': active, 'percent': 100}],
                             latestCreatedRevisionName=CANDIDATE if self.staged else self.FAILED if self.dirty else self.BASE,
                             latestReadyRevisionName=CANDIDATE if self.staged else self.BASE,
                             url='https://service.example.run.app')
        if self.dirty and not self.staged:
            obj['status']['conditions'][0].update(status='False', reason='SECRETS_ACCESS_CHECK_FAILED')
        if self.staged:
            obj['status']['traffic'].append({'revisionName': CANDIDATE, 'percent': 0,
                'tag': 'p6-candidate', 'url': 'https://candidate.example.run.app'})
            if self.fault == 'traffic':
                obj['status']['traffic'][0]['revisionName'] = CANDIDATE
                obj['spec']['traffic'][0]['revisionName'] = CANDIDATE
            if self.fault == 'config':
                obj['spec']['template']['spec']['timeoutSeconds'] += 1
            if self.fault == 'annotations':
                obj['spec']['template']['metadata']['annotations']['autoscaling.knative.dev/maxScale'] = '99'
            if self.fault == 'service_ingress':
                obj['metadata']['annotations']['run.googleapis.com/ingress'] = 'internal'
            if self.fault == 'late_usage' and 'health' in self.events:
                obj['spec']['template']['spec']['containers'][0]['env'].append(
                    {'name': 'MM_USAGE_ENFORCEMENT', 'value': 'required'})
        return obj

    def gcloud(self, *args):
        self.commands.append(args)
        if args[:2] == ('run', 'deploy'):
            if self.fault == 'deploy':
                raise guard.GuardError('GCLOUD_COMMAND_FAILED')
            spec = copy.deepcopy(self.failed_spec if self.dirty else self.before_spec)
            container = spec['containers'][0]
            env = {row['name']: row for row in container['env']}
            for arg in args:
                if arg.startswith('--update-env-vars='):
                    for pair in arg.split('=', 1)[1].split(','):
                        name, value = pair.split('=', 1)
                        env[name] = {'name': name, 'value': value}
                elif arg.startswith(('--remove-env-vars=', '--remove-secrets=')):
                    for name in arg.split('=', 1)[1].split(','):
                        env.pop(name, None)
                elif arg.startswith('--image='):
                    container['image'] = arg.split('=', 1)[1]
            container['env'] = list(env.values())
            if self.fault == 'usage_reintroduced':
                container['env'].append({'name': 'MM_USAGE_ENFORCEMENT', 'value': 'required'})
            self.candidate_spec = spec
            self.staged = True
            self.events.append('stage')
            return ''
        if args[:3] == ('run', 'services', 'describe'):
            return json.dumps(self.service())
        if args[:3] == ('run', 'revisions', 'describe'):
            if args[3] == self.BASE:
                return json.dumps(self.active_revision())
            if args[3] != CANDIDATE:
                raise AssertionError('Unexpected revision read')
            obj = ready_object(CANDIDATE, copy.deepcopy(self.candidate_spec))
            obj['metadata']['annotations'] = {**self.annotations, 'run.googleapis.com/ingress': 'all'}
            obj['status']['imageDigest'] = guard.IMAGE_ROOT + '@' + DIGEST
            if self.fault == 'not_ready':
                obj['status']['conditions'][0]['status'] = 'False'
            if self.fault == 'revision_annotations':
                obj['metadata']['annotations']['run.googleapis.com/cloudsql-instances'] = 'wrong'
            if self.fault == 'revision_ingress':
                obj['metadata']['annotations']['run.googleapis.com/ingress'] = 'internal'
            return json.dumps(obj)
        if args[:3] == ('run', 'revisions', 'list'):
            names = [self.BASE, self.FAILED] + ([CANDIDATE] if self.staged else [])
            if self.staged and self.fault == 'concurrent_revision':
                names.append(guard.SERVICE + '-concurrent')
            return json.dumps([{'metadata': {'name': name}} for name in names])
        if args[:3] == ('run', 'services', 'update-traffic'):
            self.traffic_revision = next(arg for arg in args if arg.startswith('--to-revisions=')).split('=', 2)[1]
            self.promoted = self.traffic_revision == CANDIDATE
            self.events.append('promote')
            return ''
        if args[:4] == ('artifacts', 'docker', 'images', 'describe'):
            return DIGEST
        raise AssertionError('Unexpected command in offline quality test')

    def health(self, url, commit, revision, **kwargs):
        self.events.append('health')
        if self.fault == 'health':
            raise guard.GuardError('HEALTH_NOT_OK')
        assert (commit, revision) in {(COMMIT, CANDIDATE), (self.SHA, self.BASE)}
        assert not kwargs


class QualityReleaseTests(unittest.TestCase):
    def execute(self, cloud, expected=QualityCloud.SHA, *, operation='deploy'):
        with patch.object(guard, 'gcloud', side_effect=cloud.gcloud), \
             patch.object(guard, 'health', side_effect=cloud.health), \
             contextlib.redirect_stdout(io.StringIO()):
            return guard.execute(guard.PROJECT, COMMIT, BUILD, operation, expected, preserve_active=True,
                                 expected_current_revision=cloud.BASE, expected_current_digest=cloud.BASE_DIGEST)

    def preflight(self, service, revision):
        return guard.quality_preflight(service, revision, expected_current=QualityCloud.SHA,
                                       expected_current_revision=QualityCloud.BASE,
                                       expected_current_digest=QualityCloud.BASE_DIGEST)

    def test_preflight_uses_only_control_plane_reads_without_image_or_health_calls(self):
        allowed = {('run', 'services', 'describe'), ('run', 'revisions', 'describe'),
                   ('run', 'revisions', 'list')}
        for dirty in (False, True):
            with self.subTest(dirty=dirty):
                cloud = QualityCloud(dirty=dirty)

                def read_only_gcloud(*args):
                    self.assertIn(args[:3], allowed)
                    return cloud.gcloud(*args)

                with patch.object(guard, 'gcloud', side_effect=read_only_gcloud), \
                     patch.object(guard, 'health', side_effect=AssertionError('No service requests')), \
                     patch.object(guard.urllib.request, 'build_opener', side_effect=AssertionError('No service HTTP')), \
                     contextlib.redirect_stdout(io.StringIO()):
                    result = guard.execute(guard.PROJECT, COMMIT, BUILD, 'preflight', cloud.SHA,
                                           preserve_active=True, expected_current_revision=cloud.BASE,
                                           expected_current_digest=cloud.BASE_DIGEST)
                self.assertEqual(result['status'], 'PASS_REQUIRED_PREFLIGHT')
                self.assertEqual(result['revision'], cloud.BASE)
                self.assertEqual(result['runtime_commit_sha'], cloud.SHA)
                self.assertEqual(result['image_digest'], cloud.BASE_DIGEST)
                self.assertEqual(result['known_usage_cleanup_required'], dirty)
                self.assertFalse(result['new_revision'])
                self.assertEqual(cloud.events, [])
                self.assertEqual([args[:3] for args in cloud.commands],
                                 [('run', 'services', 'describe'), ('run', 'revisions', 'describe'),
                                  ('run', 'revisions', 'list')])

    def test_preflight_stale_identity_or_configuration_fails_without_mutation(self):
        for fault, code in (('sha', 'RUNTIME_SHA_MISMATCH'),
                            ('authority', 'AUTHORITY_MUST_REMAIN_REQUIRED'),
                            ('usage', 'QUALITY_ACTIVE_USAGE_CONFIGURATION_CHANGED')):
            with self.subTest(fault=fault):
                cloud = QualityCloud(dirty=False)
                if fault == 'sha':
                    cloud.before_spec['containers'][0]['env'][0]['value'] = 'f' * 40
                elif fault == 'authority':
                    cloud.before_spec['containers'][0]['env'][1]['value'] = 'off'
                else:
                    cloud.before_spec['containers'][0]['env'].append(
                        {'name': 'MM_USAGE_ENFORCEMENT', 'value': 'required'})
                with self.assertRaisesRegex(guard.GuardError, '^' + code + '$'):
                    self.execute(cloud, operation='preflight')
                self.assertEqual(cloud.events, [])
                self.assertTrue(all(args[:3] in {('run', 'services', 'describe'),
                                               ('run', 'revisions', 'describe')}
                                    for args in cloud.commands))

    def test_successful_preflight_does_not_bypass_fresh_deploy_identity_check(self):
        cloud = QualityCloud(dirty=False)
        self.execute(cloud, operation='preflight')
        cloud.before_spec['containers'][0]['env'][0]['value'] = 'f' * 40
        cloud.commands.clear()
        with self.assertRaisesRegex(guard.GuardError, '^RUNTIME_SHA_MISMATCH$'):
            self.execute(cloud)
        self.assertEqual(cloud.events, [])
        self.assertEqual([args[:3] for args in cloud.commands],
                         [('run', 'services', 'describe'), ('run', 'revisions', 'describe')])

    def test_preflight_cannot_use_less_restrictive_legacy_mode(self):
        with patch.object(guard, 'gcloud', side_effect=AssertionError('Must fail before I/O')):
            with self.assertRaisesRegex(guard.GuardError, '^PREFLIGHT_REQUIRES_ACTIVE_PRESERVATION$'):
                guard.execute(guard.PROJECT, COMMIT, BUILD, 'preflight', QualityCloud.SHA)

    def test_stage_returns_verified_candidate_and_previous_identity_at_zero_traffic(self):
        class CurrentQualityCloud(QualityCloud):
            BASE = guard.SERVICE + '-current-verified'
            SHA = '3' * 40
            BASE_DIGEST = 'sha256:' + 'c' * 64

        for cloud in (QualityCloud(), CurrentQualityCloud(dirty=False)):
            with self.subTest(previous_revision=cloud.BASE):
                result = self.execute(cloud, expected=cloud.SHA, operation='stage')
                self.assertEqual(result['status'], 'PASS_REQUIRED_STAGED')
                self.assertEqual(result['revision'], CANDIDATE)
                self.assertEqual(result['runtime_commit_sha'], COMMIT)
                self.assertEqual(result['image_digest'], DIGEST)
                self.assertEqual(result['candidate_url'], 'https://candidate.example.run.app')
                self.assertEqual(result['candidate_traffic_percent'], 0)
                self.assertEqual(result['previous_revision'], cloud.BASE)
                self.assertEqual(result['previous_runtime_commit_sha'], cloud.SHA)
                self.assertEqual(result['previous_image_digest'], cloud.BASE_DIGEST)
                self.assertEqual(result['previous_traffic_percent'], 100)
                self.assertTrue(result['active_configuration_preserved'])
                self.assertTrue(result['candidate_health_verified'])
                self.assertFalse(result['promoted'])
                self.assertEqual(cloud.traffic_revision, cloud.BASE)
                self.assertEqual(cloud.events, ['stage', 'health'])
                self.assertEqual(guard.PHASE, 'candidate_ready_no_traffic')
                self.assertFalse(any(args[:3] == ('run', 'services', 'update-traffic')
                                     for args in cloud.commands))

    def test_stage_rechecks_candidate_tag_after_health_without_promoting(self):
        for fault in ('missing', 'other_revision', 'other_url', 'duplicate'):
            with self.subTest(fault=fault):
                class ChangedTagCloud(QualityCloud):
                    def service(self):
                        obj = super().service()
                        if self.staged and 'health' in self.events:
                            tags = obj['status']['traffic']
                            if fault == 'missing':
                                tags.pop()
                            elif fault == 'other_revision':
                                tags[-1]['revisionName'] = self.BASE
                            elif fault == 'other_url':
                                tags[-1]['url'] = 'https://other.example.run.app'
                            else:
                                tags.append(dict(tags[-1]))
                        return obj

                cloud = ChangedTagCloud(dirty=False)
                with self.assertRaisesRegex(guard.GuardError, '^CANDIDATE_TAG_CHANGED$'):
                    self.execute(cloud, operation='stage')
                self.assertEqual(cloud.events, ['stage', 'health'])
                self.assertFalse(any(args[:3] == ('run', 'services', 'update-traffic')
                                     for args in cloud.commands))

    def test_stage_validation_faults_fail_without_promoting(self):
        for fault in ('deploy', 'traffic', 'config', 'annotations', 'revision_annotations',
                      'revision_ingress', 'service_ingress', 'not_ready', 'concurrent_revision',
                      'usage_reintroduced', 'late_usage', 'health'):
            with self.subTest(fault=fault):
                cloud = QualityCloud(fault)
                with self.assertRaises(guard.GuardError):
                    self.execute(cloud, operation='stage')
                self.assertFalse(any(args[:3] == ('run', 'services', 'update-traffic')
                                     for args in cloud.commands))

    def test_failed_latest_is_restored_from_ready_active_without_usage_migration(self):
        cloud = QualityCloud()
        result = self.execute(cloud)
        self.assertEqual(result['previous_revision'], cloud.BASE)
        self.assertTrue(result['active_configuration_preserved'])
        self.assertEqual(cloud.events, ['stage', 'health', 'promote', 'health'])
        deploy = next(args for args in cloud.commands if args[:2] == ('run', 'deploy'))
        self.assertIn('--no-traffic', deploy)
        self.assertIn('--update-env-vars=COMMIT_SHA=' + COMMIT, deploy)
        self.assertIn('--remove-env-vars=MM_USAGE_ENFORCEMENT,MM_USAGE_TIMEZONE', deploy)
        self.assertIn('--remove-secrets=MM_USAGE_AUTHORITY_SECRET', deploy)
        self.assertFalse(any(arg.startswith('--update-secrets=') for arg in deploy))
        self.assertFalse(guard.USAGE_ENV & guard.environment(cloud.candidate_spec).keys())
        self.assertEqual(guard.unaffected_spec(cloud.candidate_spec), guard.unaffected_spec(cloud.before_spec))

    def test_clean_template_needs_no_environment_removal(self):
        cloud = QualityCloud(dirty=False)
        self.execute(cloud)
        deploy = next(args for args in cloud.commands if args[:2] == ('run', 'deploy'))
        self.assertFalse(any(arg.startswith(('--remove-env-vars=', '--remove-secrets=')) for arg in deploy))

    def test_quality_preflight_checks_identity_and_effective_traffic_without_io(self):
        cloud = QualityCloud()
        before = cloud.service()
        revision = cloud.active_revision()
        mutations = {
            'active_sha': lambda s, r: r['spec']['containers'][0]['env'][0].update(value='f' * 40),
            'active_name': lambda s, r: r['metadata'].update(name=guard.SERVICE + '-other'),
            'active_digest': lambda s, r: r['status'].update(imageDigest='sha256:' + 'f' * 64),
            'active_not_ready': lambda s, r: r['status']['conditions'][0].update(status='False'),
            'unobserved': lambda s, r: s['status'].update(observedGeneration=0),
            'different_requested': lambda s, r: s['spec']['traffic'][0].update(revisionName=guard.SERVICE + '-other'),
            'moving_traffic': lambda s, r: s['spec']['traffic'][0].update(latestRevision=True),
            'unknown_failed_revision': lambda s, r: s['status'].update(latestCreatedRevisionName=guard.SERVICE + '-unknown'),
            'unknown_failed_sha': lambda s, r: s['spec']['template']['spec']['containers'][0]['env'][0].update(value='e' * 40),
            'unknown_failed_image': lambda s, r: s['spec']['template']['spec']['containers'][0].update(image='unexpected'),
            'template_annotation': lambda s, r: s['spec']['template']['metadata']['annotations'].update({'autoscaling.knative.dev/maxScale': '99'}),
            'active_ingress': lambda s, r: r['metadata']['annotations'].update({'run.googleapis.com/ingress': 'internal'}),
            'service_ingress': lambda s, r: s['metadata']['annotations'].update({'run.googleapis.com/ingress': 'internal'}),
            'service_account': lambda s, r: s['spec']['template']['spec'].update(serviceAccountName='other'),
            'authority_changed': lambda s, r: s['spec']['template']['spec']['containers'][0]['env'][1].update(value='off'),
        }
        with patch.object(guard, 'gcloud', side_effect=AssertionError('No cloud calls in preflight')):
            # The historical preflight rejected the whole service solely because
            # latestCreated failed, although the fixed serving revision is Ready.
            with self.assertRaisesRegex(guard.GuardError, '^RESOURCE_NOT_READY$'):
                guard.pinned_traffic(before)
            self.assertTrue(self.preflight(before, revision))
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    service, active = copy.deepcopy(before), copy.deepcopy(revision)
                    mutate(service, active)
                    with self.assertRaises(guard.GuardError):
                        self.preflight(service, active)

    def test_active_usage_configuration_is_never_downgraded(self):
        for row in (
            {'name': 'MM_USAGE_ENFORCEMENT', 'value': 'required'},
            {'name': 'MM_USAGE_ENFORCEMENT', 'value': 'off'},
            {'name': 'MM_USAGE_AUTHORITY_SECRET', 'valueFrom': {'secretKeyRef': dict(guard.USAGE_SECRET)}},
        ):
            with self.subTest(name=row['name'], value=row.get('value')):
                cloud = QualityCloud()
                cloud.before_spec['containers'][0]['env'].append(row)
                with self.assertRaises(guard.GuardError):
                    self.execute(cloud)
                self.assertNotIn('stage', cloud.events)

    def test_unexpected_failed_usage_configuration_stops_before_stage(self):
        for mutation in ('partial', 'timezone', 'secret'):
            with self.subTest(mutation=mutation):
                cloud = QualityCloud()
                env = cloud.failed_spec['containers'][0]['env']
                if mutation == 'partial':
                    env.pop()
                elif mutation == 'timezone':
                    next(r for r in env if r['name'] == 'MM_USAGE_TIMEZONE')['value'] = 'UTC'
                else:
                    next(r for r in env if r['name'] == 'MM_USAGE_AUTHORITY_SECRET')['valueFrom']['secretKeyRef']['key'] = '999'
                with self.assertRaises(guard.GuardError):
                    self.execute(cloud)
                self.assertNotIn('stage', cloud.events)

    def test_wrong_expected_baseline_stops_before_stage(self):
        cloud = QualityCloud()
        with self.assertRaises(guard.GuardError):
            self.execute(cloud, expected='f' * 40)
        self.assertNotIn('stage', cloud.events)

    def test_candidate_or_validation_faults_cannot_promote(self):
        for fault in ('deploy', 'traffic', 'config', 'annotations', 'revision_annotations', 'revision_ingress', 'service_ingress',
                      'not_ready', 'concurrent_revision', 'usage_reintroduced', 'late_usage', 'health'):
            with self.subTest(fault=fault):
                cloud = QualityCloud(fault)
                with self.assertRaises(guard.GuardError):
                    self.execute(cloud)
                self.assertNotIn('promote', cloud.events)

    def test_quality_cannot_enable_legacy_usage_branch(self):
        with patch.object(guard, 'gcloud', side_effect=AssertionError('Must fail before I/O')):
            with self.assertRaisesRegex(guard.GuardError, '^QUALITY_USAGE_MIGRATION_FORBIDDEN$'):
                guard.execute(guard.PROJECT, COMMIT, BUILD, 'deploy', QualityCloud.SHA,
                              preserve_active=True, usage_required=True)

    def test_quality_rollback_only_routes_to_verified_baseline_without_new_revision(self):
        cloud = QualityCloud(dirty=False)
        cloud.gcloud('run', 'deploy', guard.SERVICE, '--image=' + guard.IMAGE_ROOT + '@' + DIGEST,
                     '--update-env-vars=COMMIT_SHA=' + COMMIT)
        cloud.traffic_revision = CANDIDATE
        cloud.commands.clear()
        cloud.events.clear()
        with patch.object(guard, 'gcloud', side_effect=cloud.gcloud), \
             patch.object(guard, 'health', side_effect=cloud.health), \
             contextlib.redirect_stdout(io.StringIO()):
            result = guard.execute(guard.PROJECT, COMMIT, BUILD, 'rollback', COMMIT, preserve_active=True,
                                   expected_current_revision=CANDIDATE, expected_current_digest=DIGEST)
        self.assertEqual(result['revision'], cloud.BASE)
        self.assertEqual(result['runtime_commit_sha'], cloud.SHA)
        self.assertEqual(result['image_digest'], cloud.BASE_DIGEST)
        self.assertFalse(result['new_revision'])
        self.assertNotIn('stage', cloud.events)
        updates = [args for args in cloud.commands if args[:3] == ('run', 'services', 'update-traffic')]
        self.assertEqual(len(updates), 1)
        self.assertIn('--to-revisions=' + cloud.BASE + '=100', updates[0])

    def test_main_uses_quality_preservation_and_pinned_rollback_baseline(self):
        self.assertEqual(guard.BASELINE_REVISION, QualityCloud.BASE)
        self.assertEqual(guard.BASELINE_COMMIT, QualityCloud.SHA)
        self.assertEqual(guard.BASELINE_DIGEST, QualityCloud.BASE_DIGEST)
        env = {'MM_P6_EXPECTED_CURRENT_SHA': QualityCloud.SHA,
               'MM_P6_EXPECTED_CURRENT_REVISION': QualityCloud.BASE,
               'MM_P6_EXPECTED_CURRENT_DIGEST': QualityCloud.BASE_DIGEST}
        with patch.dict(guard.os.environ, env, clear=True), \
             patch.object(guard, 'execute', return_value={}) as execute, \
             contextlib.redirect_stdout(io.StringIO()):
            guard.main()
        self.assertEqual(execute.call_args.kwargs, {'preserve_active': True,
                         'expected_current_revision': QualityCloud.BASE,
                         'expected_current_digest': QualityCloud.BASE_DIGEST})
        self.assertEqual(execute.call_args.args[4], QualityCloud.SHA)

    def test_next_release_accepts_explicit_verified_current_identity(self):
        class LaterQualityCloud(QualityCloud):
            BASE = guard.SERVICE + '-later-verified'
            SHA = '3' * 40
            BASE_DIGEST = 'sha256:' + 'c' * 64

        cloud = LaterQualityCloud(dirty=False)
        result = self.execute(cloud, expected=cloud.SHA)
        self.assertEqual(result['previous_revision'], cloud.BASE)
        self.assertEqual(cloud.events, ['stage', 'health', 'promote', 'health'])
        self.assertEqual(guard.unaffected_spec(cloud.before_spec), guard.unaffected_spec(cloud.candidate_spec))
        deploy = next(args for args in cloud.commands if args[:2] == ('run', 'deploy'))
        self.assertFalse(any(arg.startswith(('--remove-env-vars=', '--remove-secrets=', '--update-secrets='))
                             for arg in deploy))

    def test_expected_identity_inputs_are_required_before_any_cloud_call(self):
        defaults = {'expected_current': QualityCloud.SHA,
                    'expected_current_revision': QualityCloud.BASE,
                    'expected_current_digest': QualityCloud.BASE_DIGEST}
        cases = (
            ({'expected_current': None}, 'EXPECTED_CURRENT_SHA_INVALID'),
            ({'expected_current': 'a' * 39}, 'EXPECTED_CURRENT_SHA_INVALID'),
            ({'expected_current': 'A' * 40}, 'EXPECTED_CURRENT_SHA_INVALID'),
            ({'expected_current': 'z' * 40}, 'EXPECTED_CURRENT_SHA_INVALID'),
            ({'expected_current_revision': None}, 'EXPECTED_CURRENT_REVISION_INVALID'),
            ({'expected_current_revision': ''}, 'EXPECTED_CURRENT_REVISION_INVALID'),
            ({'expected_current_revision': 'other-service-b123'}, 'EXPECTED_CURRENT_REVISION_INVALID'),
            ({'expected_current_revision': QualityCloud.BASE + '/bad'}, 'EXPECTED_CURRENT_REVISION_INVALID'),
            ({'expected_current_digest': None}, 'EXPECTED_CURRENT_DIGEST_INVALID'),
            ({'expected_current_digest': ''}, 'EXPECTED_CURRENT_DIGEST_INVALID'),
            ({'expected_current_digest': 'sha256:not-a-digest'}, 'EXPECTED_CURRENT_DIGEST_INVALID'),
        )
        with patch.object(guard, 'gcloud', side_effect=AssertionError('Must fail before I/O')):
            for changed, code in cases:
                with self.subTest(changed=changed), self.assertRaisesRegex(guard.GuardError, '^' + code + '$'):
                    guard.execute(guard.PROJECT, COMMIT, BUILD, 'deploy', preserve_active=True,
                                  **{**defaults, **changed})

    def test_current_identity_mismatches_stop_before_stage(self):
        cases = (
            ({'expected_current': 'f' * 40}, 'RUNTIME_SHA_MISMATCH'),
            ({'expected_current_revision': guard.SERVICE + '-unexpected'}, 'EXPECTED_CURRENT_REVISION_MISMATCH'),
            ({'expected_current_digest': 'sha256:' + 'f' * 64}, 'EXPECTED_CURRENT_DIGEST_MISMATCH'),
        )
        for changed, code in cases:
            cloud = QualityCloud(dirty=False)
            expected = {'expected_current': cloud.SHA, 'expected_current_revision': cloud.BASE,
                        'expected_current_digest': cloud.BASE_DIGEST}
            with self.subTest(changed=changed), \
                 patch.object(guard, 'gcloud', side_effect=cloud.gcloud), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 self.assertRaisesRegex(guard.GuardError, '^' + code + '$'):
                guard.execute(guard.PROJECT, COMMIT, BUILD, 'deploy', preserve_active=True,
                              **{**expected, **changed})
            self.assertNotIn('stage', cloud.events)

    def test_expected_digest_requires_matching_immutable_image_reference(self):
        cloud = QualityCloud(dirty=False)
        cloud.before_spec['containers'][0]['image'] = guard.IMAGE_ROOT + ':mutable-tag'
        with self.assertRaisesRegex(guard.GuardError, '^EXPECTED_CURRENT_IMAGE_MISMATCH$'):
            self.execute(cloud)
        self.assertNotIn('stage', cloud.events)

    def test_failed_template_cleanup_remains_limited_to_historical_baseline(self):
        class LaterQualityCloud(QualityCloud):
            BASE = guard.SERVICE + '-later-verified'
            SHA = '3' * 40
            BASE_DIGEST = 'sha256:' + 'c' * 64

        cloud = LaterQualityCloud(dirty=True)
        with self.assertRaisesRegex(guard.GuardError, '^QUALITY_CLEANUP_BASELINE_MISMATCH$'):
            self.execute(cloud, expected=cloud.SHA)
        self.assertNotIn('stage', cloud.events)


class HealthSmokeTests(unittest.TestCase):
    def payloads(self):
        return {
            '/ping': {'ok': True},
            '/version': {'ok': True, 'commit_sha': COMMIT, 'revision': CANDIDATE},
            '/openapi.json': {'openapi': '3.1.0', 'paths': {
                path: {'post': {'responses': {'200': {'description': 'OK'}}}}
                for path in guard.OPENAPI_POST_PATHS
            }},
        }

    def run_health(self, payloads, *, status=200):
        requests = []
        handlers = []

        class Response:
            def __init__(self, body):
                self.body = body
                self.status = status
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self, maximum):
                return self.body[:maximum]

        class Opener:
            def open(self, request, timeout):
                requests.append(request)
                path = guard.urllib.parse.urlsplit(request.full_url).path
                payload = payloads[path]
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                return Response(raw)

        def build_opener(handler):
            handlers.append(handler() if isinstance(handler, type) else handler)
            return Opener()

        with patch.object(guard.urllib.request, 'build_opener', side_effect=build_opener):
            guard.health('https://candidate.example.run.app', COMMIT, CANDIDATE)
        return requests, handlers

    def test_identity_and_five_post_contracts_use_only_bounded_get_requests(self):
        payloads = self.payloads()
        payloads['/openapi.json']['info'] = {'description': 'x' * 300000}
        requests, handlers = self.run_health(payloads)
        self.assertEqual([r.get_method() for r in requests], ['GET', 'GET', 'GET'])
        self.assertEqual([guard.urllib.parse.urlsplit(r.full_url).path for r in requests],
                         ['/ping', '/version', '/openapi.json'])
        self.assertEqual(len(handlers), 1)
        self.assertIsNone(handlers[0].redirect_request(None, None, 302, '', {}, 'https://elsewhere.example'))

    def test_each_missing_or_wrong_method_blocks_smoke(self):
        for path in guard.OPENAPI_POST_PATHS:
            for mutation in ('missing', 'get_only', 'empty_operation'):
                with self.subTest(path=path, mutation=mutation):
                    payloads = self.payloads()
                    paths = payloads['/openapi.json']['paths']
                    if mutation == 'missing':
                        paths.pop(path)
                    elif mutation == 'get_only':
                        paths[path] = {'get': paths[path]['post']}
                    else:
                        paths[path] = {'post': {}}
                    with self.assertRaisesRegex(guard.GuardError, '^OPENAPI_OPERATION_MISSING$'):
                        self.run_health(payloads)

    def test_invalid_or_oversized_schema_and_non_success_http_stop_smoke(self):
        cases = (
            ({'openapi': '3.1.0', 'paths': []}, 'OPENAPI_SCHEMA_INVALID'),
            ([], 'HEALTH_NOT_OBJECT'),
            (b'not JSON', 'HEALTH_REQUEST_FAILED'),
            (b'x' * (guard.OPENAPI_MAX_BYTES + 1), 'HEALTH_HTTP_INVALID'),
        )
        for schema, code in cases:
            with self.subTest(code=code):
                payloads = self.payloads()
                payloads['/openapi.json'] = schema
                with self.assertRaisesRegex(guard.GuardError, '^' + code + '$'):
                    self.run_health(payloads)
        for status in (302, 403, 500):
            with self.subTest(status=status):
                with self.assertRaisesRegex(guard.GuardError, '^HEALTH_HTTP_INVALID$'):
                    self.run_health(self.payloads(), status=status)

    def test_wrong_revision_identity_and_non_run_app_url_are_rejected(self):
        payloads = self.payloads()
        payloads['/version']['revision'] = QualityCloud.BASE
        with self.assertRaisesRegex(guard.GuardError, '^HEALTH_IDENTITY_MISMATCH$'):
            self.run_health(payloads)
        with patch.object(guard.urllib.request, 'build_opener', side_effect=AssertionError('No outbound call')):
            for url in ('http://service.example.run.app', 'https://example.org',
                        'https://service.example.run.app/redirect', 'https://service.example.run.app?next=elsewhere'):
                with self.subTest(url=url):
                    with self.assertRaisesRegex(guard.GuardError, '^HEALTH_URL_INVALID$'):
                        guard.health(url, COMMIT, CANDIDATE)


class DiagnosticsTests(unittest.TestCase):
    def test_failed_command_reports_only_safe_metadata(self):
        marker = 'SECRET_MUST_NEVER_APPEAR'
        result = subprocess.CompletedProcess(['gcloud'], 7, marker,
                    'PERMISSION_DENIED: ' + marker)
        with patch.object(guard.subprocess, 'run', return_value=result):
            with self.assertRaises(guard.GuardError) as caught:
                guard.gcloud('run', 'deploy', marker, '--update-env-vars=' + marker)
        error = caught.exception
        self.assertEqual(error.diagnostics, {'command': 'run.deploy', 'returncode': 7,
                                             'category': 'permission_denied'})
        with patch.object(guard, 'execute', side_effect=error), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as exited:
                guard.main()
        self.assertEqual(exited.exception.code, 2)
        report = json.loads(output.getvalue())
        self.assertEqual(report['code'], 'GCLOUD_COMMAND_FAILED')
        self.assertEqual(report['command'], 'run.deploy')
        self.assertNotIn(marker, output.getvalue())

    def test_timeout_and_process_failure_are_distinguished_without_raw_output(self):
        failures = (
            (subprocess.TimeoutExpired(['SECRET'], 900, output='SECRET', stderr='SECRET'), 'process_timeout'),
            (OSError('SECRET'), 'process_unavailable'),
        )
        for failure, category in failures:
            with self.subTest(category=category), \
                 patch.object(guard.subprocess, 'run', side_effect=failure):
                with self.assertRaises(guard.GuardError) as caught:
                    guard.gcloud('run', 'deploy', 'SECRET')
                self.assertEqual(caught.exception.diagnostics,
                                 {'command': 'run.deploy', 'category': category})
                self.assertNotIn('SECRET', str(caught.exception))

    def test_unknown_command_or_error_cannot_echo_untrusted_text(self):
        self.assertEqual(guard.gcloud_command_family(('SECRET', 'deploy')), 'unclassified')
        self.assertEqual(guard.gcloud_failure_category('SECRET'), 'unclassified')


if __name__ == '__main__':
    unittest.main()
