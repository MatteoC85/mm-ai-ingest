"""Quality deployment through the GitHub-triggered Cloud Build.

Read-only preflight -> new image -> no traffic -> config/health. Stage stops here;
the legacy deploy operation continues to exact promotion.
Preserve the verified active configuration and ASK authority. No provider, secret
retrieval or schema writes. Never invoke manually in Cloud Shell. Import is inert.
"""
from __future__ import annotations
import copy
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request

PROJECT = 'machinemind-ai-2a'
REGION = 'europe-west1'
SERVICE = 'mm-ai-ingest-prod'
IMAGE_ROOT = f'{REGION}-docker.pkg.dev/{PROJECT}/cloud-run-source-deploy/mm-ai-ingest/{SERVICE}'
BASELINE_REVISION = 'mm-ai-ingest-prod-bc1a194bea02e4191a33773d371238af2'
BASELINE_COMMIT = '3608ee56b2eb292c35c34558e5ae998425a26504'
BASELINE_DIGEST = 'sha256:6db9b12ca70c6e11f1a6fd2d7e69fbb1a1901b62b155cf320bff82ee434f4656'
FAILED_REVISION = 'mm-ai-ingest-prod-bf992ccb586fc4cf1830480a55145a9bb'
FAILED_COMMIT = '201e9492fb5fc061f1702ec8925c051567963125'
FAILED_DIGEST = 'sha256:2ad3ff7448219e818fe705fdba4c906830b45ca163f4956602bd27c74c7b3871'
SECRET_REFS = {
    'MM_APP_AUTHORITY_SECRET': {'name': 'mm-app-authority-v1', 'key': '2'},
    'MM_BUBBLE_AUTHORITY_TOKEN': {'name': 'mm-bubble-authority-token-v1', 'key': '2'},
}
PHASE = 'initialize'
USAGE_VERSION = 'p6-interactive-ledger-v1'
USAGE_SECRET = {'name': 'mm-usage-authority-v6', 'key': '1'}
USAGE_ENV = frozenset({'MM_USAGE_ENFORCEMENT', 'MM_USAGE_TIMEZONE', 'MM_USAGE_AUTHORITY_SECRET'})
OPENAPI_POST_PATHS = (
    '/v1/ai/ask', '/v1/ai/root-cause', '/v1/ai/smart-diagnostic/start',
    '/v1/ai/smart-diagnostic/answer', '/v1/ai/smart-diagnostic/finalize',
)
OPENAPI_MAX_BYTES = 2 * 1024 * 1024


class GuardError(RuntimeError):
    def __init__(self, code, *, command=None, returncode=None, category=None):
        super().__init__(code)
        # Only caller-generated identifiers and the exit status are reportable.
        # Command arguments and subprocess output may contain credentials.
        self.diagnostics = {}
        if command is not None:
            self.diagnostics['command'] = command
        if returncode is not None:
            self.diagnostics['returncode'] = returncode
        if category is not None:
            self.diagnostics['category'] = category


def require(value, code):
    if not value:
        raise GuardError(code)


def phase(value):
    global PHASE
    PHASE = value
    print('PHASE6_RELEASE_PHASE=' + value, flush=True)


def gcloud_command_family(args):
    families = (
        ('run', 'services', 'describe'),
        ('run', 'revisions', 'describe'),
        ('run', 'revisions', 'list'),
        ('run', 'services', 'update-traffic'),
        ('run', 'deploy'),
        ('artifacts', 'docker', 'images', 'describe'),
    )
    for family in families:
        if tuple(args[:len(family)]) == family:
            return '.'.join(family)
    return 'unclassified'


def gcloud_failure_category(stderr):
    """Return a bounded hint, never the untrusted diagnostic itself."""
    diagnostic = stderr.upper() if isinstance(stderr, str) else ''
    categories = (
        ('permission_denied', ('PERMISSION_DENIED', 'PERMISSION DENIED', 'DOES NOT HAVE PERMISSION')),
        ('authentication_required', ('UNAUTHENTICATED', 'NO ACTIVE ACCOUNT', 'REAUTHENTICATION_REQUIRED')),
        ('resource_not_found', ('NOT_FOUND', 'NOT FOUND')),
        ('resource_conflict', ('ALREADY_EXISTS', 'ALREADY EXISTS')),
        ('invalid_argument', ('INVALID_ARGUMENT', 'INVALID VALUE', 'UNRECOGNIZED ARGUMENTS')),
        ('resource_exhausted', ('RESOURCE_EXHAUSTED', 'QUOTA_EXCEEDED')),
        ('deadline_exceeded', ('DEADLINE_EXCEEDED', 'TIMED OUT')),
        ('container_not_ready', ('CONTAINER FAILED TO START', 'FAILED TO START AND LISTEN')),
        ('service_unavailable', ('UNAVAILABLE',)),
    )
    for category, markers in categories:
        if any(marker in diagnostic for marker in markers):
            return category
    return 'unclassified'


def gcloud(*args):
    command = gcloud_command_family(args)
    try:
        result = subprocess.run(['gcloud', *args], text=True, capture_output=True, timeout=900)
    except subprocess.TimeoutExpired:
        raise GuardError('GCLOUD_NOT_COMPLETED', command=command,
                         category='process_timeout') from None
    except OSError:
        raise GuardError('GCLOUD_NOT_COMPLETED', command=command,
                         category='process_unavailable') from None
    if result.returncode != 0:
        raise GuardError('GCLOUD_COMMAND_FAILED', command=command,
                         returncode=result.returncode,
                         category=gcloud_failure_category(result.stderr))
    return result.stdout.strip()


def resource(kind, name):
    try:
        result = json.loads(gcloud('run', kind, 'describe', name, '--project=' + PROJECT,
                                   '--region=' + REGION, '--format=json'))
    except (ValueError, TypeError):
        raise GuardError('RESOURCE_JSON_INVALID') from None
    require(type(result) is dict, 'RESOURCE_NOT_OBJECT')
    return result


def reconciled(obj):
    generation = obj.get('metadata', {}).get('generation')
    observed = obj.get('status', {}).get('observedGeneration')
    require(type(generation) is int and generation > 0 and type(observed) is int
            and observed == generation, 'RESOURCE_NOT_RECONCILED')
    require(not obj.get('metadata', {}).get('deletionTimestamp'), 'RESOURCE_DELETING')


def ready(obj):
    reconciled(obj)
    rows = [r for r in obj.get('status', {}).get('conditions', [])
            if type(r) is dict and r.get('type') == 'Ready']
    require(len(rows) == 1 and rows[0].get('status') == 'True', 'RESOURCE_NOT_READY')


def template(service):
    spec = service.get('spec', {}).get('template', {}).get('spec')
    require(type(spec) is dict, 'TEMPLATE_MISSING')
    return spec


def environment(spec):
    containers = spec.get('containers')
    require(type(containers) is list and len(containers) == 1, 'SINGLE_CONTAINER_REQUIRED')
    values = {}
    for row in containers[0].get('env', []):
        require(type(row) is dict and type(row.get('name')) is str, 'ENV_INVALID')
        require(row['name'] not in values, 'DUPLICATE_ENV')
        values[row['name']] = row
    return values


def required_configuration(spec, commit=None):
    env = environment(spec)
    require(env.get('MM_ASK_REQUEST_AUTHORITY', {}).get('value') == 'required',
            'AUTHORITY_MUST_REMAIN_REQUIRED')
    require(env.get('MM_INGEST_LEDGER_AUTO_DDL', {}).get('value') == '0', 'AUTO_DDL_NOT_DISABLED')
    for name, expected in SECRET_REFS.items():
        row = env.get(name, {})
        require('value' not in row and row.get('valueFrom', {}).get('secretKeyRef') == expected,
                'AUTHORITY_SECRET_REFERENCE_CHANGED')
    actual = env.get('COMMIT_SHA', {}).get('value')
    require(type(actual) is str and re.fullmatch('[0-9a-f]{40}', actual), 'RUNTIME_SHA_INVALID')
    if commit is not None:
        require(actual == commit, 'RUNTIME_SHA_MISMATCH')
    return actual


def unaffected_spec(spec, usage_required=False):
    value = copy.deepcopy(spec)
    environment(value)
    container = value['containers'][0]
    container.pop('image', None)
    container['env'] = sorted((r for r in container.get('env', [])
                               if r['name'] != 'COMMIT_SHA' and not (usage_required and r['name'] in USAGE_ENV)), key=lambda r: r['name'])
    return value


def usage_configuration(spec):
    env=environment(spec)
    require(env.get('MM_USAGE_ENFORCEMENT',{}).get('value')=='required','USAGE_MUST_BE_REQUIRED')
    require(env.get('MM_USAGE_TIMEZONE',{}).get('value')=='Europe/Rome','USAGE_TIMEZONE_MISMATCH')
    row=env.get('MM_USAGE_AUTHORITY_SECRET',{})
    require('value' not in row and row.get('valueFrom',{}).get('secretKeyRef')==USAGE_SECRET,
            'USAGE_SECRET_REFERENCE_MISMATCH')


def behavioral_annotations(metadata):
    """Preserve behavioral annotations; omit only CLI diagnostic provenance."""
    ignored = {'run.googleapis.com/client-name', 'run.googleapis.com/client-version',
               'run.googleapis.com/operation-id', 'run.googleapis.com/urls',
               'serving.knative.dev/creator', 'serving.knative.dev/lastModifier'}
    rows = metadata.get('annotations', {})
    require(type(rows) is dict, 'ANNOTATIONS_INVALID')
    return {k: v for k, v in rows.items() if k not in ignored}


def routing_configuration(service):
    return (behavioral_annotations(service.get('metadata', {})),
            behavioral_annotations(service.get('spec', {}).get('template', {}).get('metadata', {})))


def revision_behavioral_annotations(revision, service):
    annotations = behavioral_annotations(revision.get('metadata', {}))
    ingress = 'run.googleapis.com/ingress'
    if ingress in annotations:
        # Cloud Run copies this service-level setting into revision metadata.
        # Keep ingress in the service snapshot; ignore no unverified difference.
        require(annotations[ingress] == behavioral_annotations(service.get('metadata', {})).get(ingress),
                'REVISION_INGRESS_DIFFERS_FROM_SERVICE')
        annotations.pop(ingress)
    return annotations


def traffic(service, section):
    rows = service.get(section, {}).get('traffic')
    require(type(rows) is list, 'TRAFFIC_INVALID')
    result = {}
    for row in rows:
        require(type(row) is dict, 'TRAFFIC_INVALID')
        percent = row.get('percent', 0)
        require(type(percent) is int and 0 <= percent <= 100, 'TRAFFIC_PERCENT_INVALID')
        if percent:
            require(not row.get('latestRevision') and not row.get('configurationName'), 'TRAFFIC_NOT_FIXED')
            name = row.get('revisionName')
            require(type(name) is str and name.startswith(SERVICE + '-'), 'TRAFFIC_REVISION_INVALID')
            result[name] = result.get(name, 0) + percent
    require(sum(result.values()) == 100 and len(result) == 1, 'TRAFFIC_NOT_STABLE_100')
    return result


def pinned_traffic(service, *, allow_failed_latest=False):
    rows = [row for row in service.get('status', {}).get('conditions', [])
            if type(row) is dict and row.get('type') == 'Ready']
    known_failed_latest = (allow_failed_latest and len(rows) == 1
        and rows[0].get('status') == 'False'
        and service.get('status', {}).get('latestCreatedRevisionName') == FAILED_REVISION)
    if known_failed_latest:
        reconciled(service)
    else:
        ready(service)
    actual, requested = traffic(service, 'status'), traffic(service, 'spec')
    require(actual == requested, 'REQUESTED_EFFECTIVE_TRAFFIC_MISMATCH')
    return next(iter(actual))


def image_digest(revision):
    digest = revision.get('status', {}).get('imageDigest', '').rsplit('@', 1)[-1]
    require(re.fullmatch('sha256:[0-9a-f]{64}', digest or ''), 'IMAGE_DIGEST_INVALID')
    return digest


def expected_current_identity(commit, revision, digest):
    """Validate the reviewed release inputs without deriving trust from live state."""
    require(type(commit) is str and re.fullmatch('[0-9a-f]{40}', commit),
            'EXPECTED_CURRENT_SHA_INVALID')
    require(type(revision) is str and len(revision) <= 63
            and re.fullmatch(re.escape(SERVICE) + r'-[a-z0-9](?:[a-z0-9-]*[a-z0-9])?', revision),
            'EXPECTED_CURRENT_REVISION_INVALID')
    require(type(digest) is str and re.fullmatch('sha256:[0-9a-f]{64}', digest),
            'EXPECTED_CURRENT_DIGEST_INVALID')


def quality_preflight(service, prior_revision, *, expected_current=None,
                      expected_current_revision=None, expected_current_digest=None):
    """Read-only check of supplied metadata; return whether known Usage must be removed.

    The only permitted cleanup is the exact failed 201 template above. The
    serving revision's absent Usage configuration is never inferred from it.
    """
    expected_current_identity(expected_current, expected_current_revision, expected_current_digest)
    prior = pinned_traffic(service, allow_failed_latest=True)
    ready(prior_revision)
    require(prior_revision.get('metadata', {}).get('name') == prior, 'ACTIVE_REVISION_MISMATCH')
    active_spec = prior_revision.get('spec', {})
    active_commit = required_configuration(active_spec, expected_current)
    require(prior == expected_current_revision, 'EXPECTED_CURRENT_REVISION_MISMATCH')
    require(image_digest(prior_revision) == expected_current_digest, 'EXPECTED_CURRENT_DIGEST_MISMATCH')
    require(active_spec['containers'][0].get('image') == IMAGE_ROOT + '@' + expected_current_digest,
            'EXPECTED_CURRENT_IMAGE_MISMATCH')
    require(not (USAGE_ENV & environment(active_spec).keys()), 'QUALITY_ACTIVE_USAGE_CONFIGURATION_CHANGED')
    current_spec = template(service)
    current_commit = required_configuration(current_spec)
    current_usage = USAGE_ENV & environment(current_spec).keys()
    if current_usage or service.get('status', {}).get('latestCreatedRevisionName') == FAILED_REVISION:
        require(prior == BASELINE_REVISION and active_commit == BASELINE_COMMIT
                and expected_current_digest == BASELINE_DIGEST, 'QUALITY_CLEANUP_BASELINE_MISMATCH')
        require(current_usage == USAGE_ENV
                and service.get('status', {}).get('latestCreatedRevisionName') == FAILED_REVISION
                and current_commit == FAILED_COMMIT
                and current_spec['containers'][0].get('image') == IMAGE_ROOT + '@' + FAILED_DIGEST,
                'QUALITY_UNVERIFIED_USAGE_TEMPLATE')
        usage_configuration(current_spec)
    require(unaffected_spec(current_spec, True) == unaffected_spec(active_spec, True),
            'SERVICE_TEMPLATE_DIFFERS_FROM_ACTIVE')
    require(routing_configuration(service)[1] == revision_behavioral_annotations(prior_revision, service),
            'SERVICE_TEMPLATE_ANNOTATIONS_DIFFER_FROM_ACTIVE')
    return bool(current_usage)


def revision_names():
    try:
        rows = json.loads(gcloud('run', 'revisions', 'list', '--service=' + SERVICE,
                                '--project=' + PROJECT, '--region=' + REGION, '--format=json'))
    except (TypeError, ValueError):
        raise GuardError('REVISION_LIST_INVALID') from None
    require(type(rows) is list, 'REVISION_LIST_INVALID')
    names = [r.get('metadata', {}).get('name') for r in rows if type(r) is dict]
    require(len(names) == len(rows) and len(set(names)) == len(names)
            and all(type(n) is str and n.startswith(SERVICE + '-') for n in names),
            'REVISION_LIST_INVALID')
    return set(names)


def health(url, commit, revision, usage_required=False):
    parsed = urllib.parse.urlsplit(url) if type(url) is str else None
    require(parsed is not None and parsed.scheme == 'https' and
            re.fullmatch(r'[a-z0-9-]+(?:\.[a-z0-9-]+)*\.run\.app', parsed.netloc or '') and
            not parsed.path and not parsed.query and not parsed.fragment, 'HEALTH_URL_INVALID')
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    for suffix in ('/ping', '/version', '/openapi.json'):
        limit = OPENAPI_MAX_BYTES if suffix == '/openapi.json' else 262144
        try:
            req = urllib.request.Request(url + suffix, headers={'Cache-Control': 'no-cache'})
            with opener.open(req, timeout=20) as response:
                raw = response.read(limit + 1)
                require(response.status == 200 and len(raw) <= limit, 'HEALTH_HTTP_INVALID')
                result = json.loads(raw)
        except GuardError:
            raise
        except Exception:
            raise GuardError('HEALTH_REQUEST_FAILED') from None
        require(type(result) is dict, 'HEALTH_NOT_OBJECT')
        if suffix == '/openapi.json':
            paths = result.get('paths')
            require(type(paths) is dict and type(result.get('openapi')) is str,
                    'OPENAPI_SCHEMA_INVALID')
            for path in OPENAPI_POST_PATHS:
                item = paths.get(path)
                operation = item.get('post') if type(item) is dict else None
                require(type(operation) is dict and type(operation.get('responses')) is dict
                        and bool(operation['responses']), 'OPENAPI_OPERATION_MISSING')
            continue
        require(result.get('ok') is True, 'HEALTH_NOT_OK')
        if suffix == '/version':
            require(result.get('commit_sha') == commit and result.get('revision') == revision,
                    'HEALTH_IDENTITY_MISMATCH')
            if usage_required:
                u=result.get('usage_v6') or {}
                require(u.get('version')==USAGE_VERSION and u.get('mode')=='required' and u.get('ready') is True and u.get('database')=='postgresql',
                        'USAGE_NATIVE_HEALTH_NOT_READY')


def validate_candidate(before_spec, service, revision, name, commit, digest, usage_required=False,
                       *, expected_annotations=None):
    ready(revision)
    require(revision.get('metadata', {}).get('name') == name, 'CANDIDATE_NAME_MISMATCH')
    if expected_annotations is not None:
        require(revision_behavioral_annotations(revision, service) == expected_annotations,
                'CANDIDATE_ANNOTATIONS_CHANGED')
    for spec in (template(service), revision.get('spec', {})):
        required_configuration(spec, commit)
        if usage_required:usage_configuration(spec)
        require(unaffected_spec(spec,usage_required) == unaffected_spec(before_spec,usage_required), 'UNRELATED_CONFIG_CHANGED')
        require(spec['containers'][0].get('image') == IMAGE_ROOT + '@' + digest, 'CANDIDATE_IMAGE_NOT_PINNED')
    require(image_digest(revision) == digest, 'CANDIDATE_DIGEST_MISMATCH')


def execute(project, commit, build, operation, expected_current, *, usage_required=False, preserve_active=False,
            expected_current_revision=None, expected_current_digest=None):
    require(project == PROJECT, 'WRONG_PROJECT')
    require(type(commit) is str and re.fullmatch('[0-9a-f]{40}', commit), 'COMMIT_SHA_INVALID')
    require(type(expected_current) is str and re.fullmatch('[0-9a-f]{40}', expected_current),
            'EXPECTED_CURRENT_SHA_INVALID')
    require(type(build) is str and re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', build),
            'BUILD_ID_INVALID')
    require(operation in {'preflight', 'stage', 'deploy', 'rollback'}, 'OPERATION_INVALID')
    require(operation != 'preflight' or preserve_active, 'PREFLIGHT_REQUIRES_ACTIVE_PRESERVATION')
    require(not (preserve_active and usage_required), 'QUALITY_USAGE_MIGRATION_FORBIDDEN')
    require(not usage_required or operation in {'stage', 'deploy'}, 'LEGACY_ROLLBACK_REQUIRES_MAINTENANCE')
    if preserve_active:
        expected_current_identity(expected_current, expected_current_revision, expected_current_digest)
    phase('validate_current_required')
    before = resource('services', SERVICE)
    prior = pinned_traffic(before, allow_failed_latest=preserve_active)
    prior_revision = resource('revisions', prior)
    ready(prior_revision)
    required_configuration(prior_revision.get('spec', {}), expected_current)
    # A prior no-traffic stage, or a traffic-only rollback, can leave a newer
    # service template. The ACTIVE revision is the runtime identity. Permit
    # only image/COMMIT differences; all other settings remain identical.
    remove_usage = False
    if preserve_active:
        remove_usage = quality_preflight(before, prior_revision, expected_current=expected_current,
                                        expected_current_revision=expected_current_revision,
                                        expected_current_digest=expected_current_digest)
    else:
        required_configuration(template(before))
        require(unaffected_spec(template(before),usage_required) == unaffected_spec(prior_revision.get('spec', {}),usage_required),
                'SERVICE_TEMPLATE_DIFFERS_FROM_ACTIVE')
    validation_spec = prior_revision.get('spec', {}) if preserve_active else template(before)
    validation_options = ({'expected_annotations': revision_behavioral_annotations(prior_revision, before)}
                          if preserve_active else {})
    if expected_current == BASELINE_COMMIT:
        require(prior == BASELINE_REVISION and image_digest(prior_revision) == BASELINE_DIGEST,
                'BASELINE_IDENTITY_MISMATCH')
    names_before = revision_names()
    routing_before = routing_configuration(before)
    require(prior in names_before, 'PRIOR_REVISION_MISSING')
    if operation == 'preflight':
        # Only control-plane reads above. Deployment repeats these checks after
        # building, so this result never authorizes promotion from stale state.
        phase('read_only_preflight_complete')
        return {'status': 'PASS_REQUIRED_PREFLIGHT', 'revision': prior,
                'runtime_commit_sha': expected_current, 'image_digest': expected_current_digest,
                'authority_mode': 'required', 'new_revision': False,
                'active_configuration_preserved': True, 'known_usage_cleanup_required': remove_usage}
    if operation == 'rollback':
        phase('validate_pinned_required_rollback')
        target = resource('revisions', BASELINE_REVISION)
        ready(target)
        required_configuration(target.get('spec', {}), BASELINE_COMMIT)
        require(image_digest(target) == BASELINE_DIGEST, 'ROLLBACK_DIGEST_MISMATCH')
        if preserve_active:
            require(unaffected_spec(target.get('spec', {})) == unaffected_spec(validation_spec)
                    and revision_behavioral_annotations(target, before) == validation_options['expected_annotations'],
                    'ROLLBACK_CONFIGURATION_DIFFERS_FROM_ACTIVE')
        require(BASELINE_REVISION in names_before, 'ROLLBACK_TARGET_MISSING')
        latest = resource('services', SERVICE)
        require(pinned_traffic(latest, allow_failed_latest=preserve_active) == prior and revision_names() == names_before,
                'CONCURRENT_CHANGE_BEFORE_ROLLBACK')
        require(unaffected_spec(template(latest)) == unaffected_spec(template(before))
                and routing_configuration(latest) == routing_before,
                'CONCURRENT_CONFIG_CHANGE_BEFORE_ROLLBACK')
        phase('rollback_required_traffic_only')
        gcloud('run', 'services', 'update-traffic', SERVICE, '--project=' + PROJECT,
               '--region=' + REGION, '--to-revisions=' + BASELINE_REVISION + '=100', '--quiet')
        final = resource('services', SERVICE)
        require(routing_configuration(final) == routing_before, 'UNRELATED_ANNOTATIONS_CHANGED')
        require(pinned_traffic(final, allow_failed_latest=preserve_active) == BASELINE_REVISION, 'ROLLBACK_TRAFFIC_MISMATCH')
        require(revision_names() == names_before, 'ROLLBACK_CREATED_REVISION')
        health(final.get('status', {}).get('url'), BASELINE_COMMIT, BASELINE_REVISION)
        return {'status': 'PASS_REQUIRED_ROLLBACK', 'revision': BASELINE_REVISION,
                'runtime_commit_sha': BASELINE_COMMIT, 'image_digest': BASELINE_DIGEST,
                'previous_revision': prior, 'authority_mode': 'required', 'new_revision': False}
    phase('resolve_new_commit_image')
    digest = gcloud('artifacts', 'docker', 'images', 'describe', IMAGE_ROOT + ':' + commit,
                    '--project=' + PROJECT, '--format=value(image_summary.digest)')
    require(re.fullmatch('sha256:[0-9a-f]{64}', digest or ''), 'REGISTRY_DIGEST_INVALID')
    require(commit != expected_current, 'NEW_COMMIT_REQUIRED')
    suffix = 'b' + build.replace('-', '')
    name = SERVICE + '-' + suffix
    require(name not in names_before, 'CANDIDATE_REVISION_ALREADY_EXISTS')
    phase('stage_no_traffic')
    gcloud('run', 'deploy', SERVICE, '--project=' + PROJECT, '--region=' + REGION,
           '--platform=managed', '--image=' + IMAGE_ROOT + '@' + digest,
           '--revision-suffix=' + suffix, '--no-traffic', '--tag=p6-candidate',
           '--update-env-vars=COMMIT_SHA=' + commit + (',MM_USAGE_ENFORCEMENT=required,MM_USAGE_TIMEZONE=Europe/Rome' if usage_required else ''),
           *(['--update-secrets=MM_USAGE_AUTHORITY_SECRET='+USAGE_SECRET['name']+':'+USAGE_SECRET['key']] if usage_required else []),
           *(['--remove-env-vars=MM_USAGE_ENFORCEMENT,MM_USAGE_TIMEZONE',
              '--remove-secrets=MM_USAGE_AUTHORITY_SECRET'] if remove_usage else []), '--quiet')
    phase('read_staged_service')
    staged = resource('services', SERVICE)
    phase('read_candidate_revision')
    candidate = resource('revisions', name)
    phase('validate_staged_traffic')
    require(pinned_traffic(staged) == prior, 'TRAFFIC_CHANGED_BEFORE_VALIDATION')
    phase('validate_staged_revision_set')
    require(revision_names() == names_before | {name}, 'CONCURRENT_OR_UNEXPECTED_REVISION')
    phase('validate_staged_configuration')
    require(routing_configuration(staged) == routing_before, 'UNRELATED_ANNOTATIONS_CHANGED')
    validate_candidate(validation_spec, staged, candidate, name, commit, digest, usage_required, **validation_options)
    tags = [r for r in staged.get('status', {}).get('traffic', [])
            if r.get('tag') == 'p6-candidate' and r.get('revisionName') == name]
    require(len(tags) == 1, 'CANDIDATE_TAG_MISSING')
    phase('check_candidate_identity_before_promotion')
    health(tags[0].get('url'), commit, name, **({'usage_required':True} if usage_required else {}))
    phase('revalidate_before_promotion')
    latest = resource('services', SERVICE)
    require(pinned_traffic(latest) == prior and revision_names() == names_before | {name},
            'CONCURRENT_CHANGE_BEFORE_PROMOTION')
    require(routing_configuration(latest) == routing_before, 'UNRELATED_ANNOTATIONS_CHANGED')
    validate_candidate(validation_spec, latest, resource('revisions', name), name, commit, digest, usage_required, **validation_options)
    if operation == 'stage':
        # Health can take time: report a usable candidate only if its verified
        # URL still routes to that exact revision after the final state checks.
        final_tags = [r for r in latest.get('status', {}).get('traffic', [])
                      if r.get('tag') == 'p6-candidate']
        require(len(final_tags) == 1 and final_tags[0].get('revisionName') == name
                and final_tags[0].get('url') == tags[0].get('url')
                and final_tags[0].get('percent', 0) == 0, 'CANDIDATE_TAG_CHANGED')
        final_traffic = traffic(latest, 'status')
        phase('candidate_ready_no_traffic')
        return {'status': 'PASS_REQUIRED_STAGED', 'revision': name,
                'runtime_commit_sha': commit, 'image_digest': digest,
                'candidate_url': final_tags[0]['url'],
                'candidate_traffic_percent': final_traffic.get(name, 0),
                'previous_revision': prior, 'previous_runtime_commit_sha': expected_current,
                'previous_image_digest': image_digest(prior_revision),
                'previous_traffic_percent': final_traffic[prior],
                'authority_mode': 'required', 'new_revision': True, 'promoted': False,
                'candidate_health_verified': True, 'unrelated_spec_preserved': True,
                'usage_required': usage_required, 'usage_native_health_verified': usage_required,
                'active_configuration_preserved': preserve_active}
    phase('promote_exact_revision')
    gcloud('run', 'services', 'update-traffic', SERVICE, '--project=' + PROJECT,
           '--region=' + REGION, '--to-revisions=' + name + '=100', '--quiet')
    phase('verify_promoted_revision')
    final = resource('services', SERVICE)
    require(pinned_traffic(final) == name, 'PROMOTION_TRAFFIC_MISMATCH')
    require(revision_names() == names_before | {name}, 'CONCURRENT_CHANGE_AFTER_PROMOTION')
    require(routing_configuration(final) == routing_before, 'UNRELATED_ANNOTATIONS_CHANGED')
    validate_candidate(validation_spec, final, resource('revisions', name), name, commit, digest, usage_required, **validation_options)
    health(final.get('status', {}).get('url'), commit, name, **({'usage_required':True} if usage_required else {}))
    return {'status': 'PASS_REQUIRED_DEPLOY', 'revision': name, 'previous_revision': prior,
            'runtime_commit_sha': commit, 'image_digest': digest, 'authority_mode': 'required',
            'new_revision': True, 'health_before_promotion': True, 'unrelated_spec_preserved': True,
            'usage_required':usage_required,'usage_native_health_verified':usage_required,
            'active_configuration_preserved':preserve_active}


def main():
    try:
        result = execute(os.environ.get('MM_CB_PROJECT'), os.environ.get('MM_CB_COMMIT'),
                         os.environ.get('MM_CB_BUILD'), os.environ.get('MM_P6_OPERATION'),
                         os.environ.get('MM_P6_EXPECTED_CURRENT_SHA'), preserve_active=True,
                         expected_current_revision=os.environ.get('MM_P6_EXPECTED_CURRENT_REVISION'),
                         expected_current_digest=os.environ.get('MM_P6_EXPECTED_CURRENT_DIGEST'))
        print(json.dumps({**result, 'ask_calls': 0, 'provider_calls': 0,
                          'automatic_rollback': False, 'guard_version': 'phase6-quality-preserve-active-v2'}, sort_keys=True))
    except GuardError as exc:
        print(json.dumps({'status': 'FAIL_REQUIRED_RELEASE', 'phase': PHASE,
                          'code': str(exc), 'automatic_rollback': False,
                          **exc.diagnostics}))
        raise SystemExit(2)
    except Exception:
        print(json.dumps({'status': 'FAIL_REQUIRED_RELEASE', 'phase': PHASE,
                          'code': 'UNEXPECTED_ERROR', 'automatic_rollback': False}))
        raise SystemExit(2)


if __name__ == '__main__':
    main()
