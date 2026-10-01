"""Required-only deployment through the GitHub-triggered Cloud Build.

New commit -> immutable image -> no traffic -> config/health -> exact promotion.
Rollback targets the pinned REQUIRED baseline, never OFF. No provider, secret
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
BASELINE_REVISION = 'mm-ai-ingest-prod-b6dbe527efc7e4379992c78d21b171ee0'
BASELINE_COMMIT = '27d0f2bd5b5ae698bc6dc0cb652f50a8832dcd2b'
BASELINE_DIGEST = 'sha256:586359d9240717106938bc9c69fce50b480d25d15bfaf7aff0304d27fce63f33'
SECRET_REFS = {
    'MM_APP_AUTHORITY_SECRET': {'name': 'mm-app-authority-v1', 'key': '2'},
    'MM_BUBBLE_AUTHORITY_TOKEN': {'name': 'mm-bubble-authority-token-v1', 'key': '2'},
}
PHASE = 'initialize'
USAGE_VERSION = 'p6-interactive-ledger-v1'
USAGE_SECRET = {'name': 'mm-usage-authority-v6', 'key': '1'}
USAGE_ENV = frozenset({'MM_USAGE_ENFORCEMENT', 'MM_USAGE_TIMEZONE', 'MM_USAGE_AUTHORITY_SECRET'})


class GuardError(RuntimeError):
    pass


def require(value, code):
    if not value:
        raise GuardError(code)


def phase(value):
    global PHASE
    PHASE = value
    print('PHASE6_RELEASE_PHASE=' + value, flush=True)


def gcloud(*args):
    try:
        result = subprocess.run(['gcloud', *args], text=True, capture_output=True, timeout=900)
    except (OSError, subprocess.TimeoutExpired):
        raise GuardError('GCLOUD_NOT_COMPLETED') from None
    require(result.returncode == 0, 'GCLOUD_COMMAND_FAILED')
    return result.stdout.strip()


def resource(kind, name):
    try:
        result = json.loads(gcloud('run', kind, 'describe', name, '--project=' + PROJECT,
                                   '--region=' + REGION, '--format=json'))
    except (ValueError, TypeError):
        raise GuardError('RESOURCE_JSON_INVALID') from None
    require(type(result) is dict, 'RESOURCE_NOT_OBJECT')
    return result


def ready(obj):
    generation = obj.get('metadata', {}).get('generation')
    observed = obj.get('status', {}).get('observedGeneration')
    require(type(generation) is int and generation > 0 and type(observed) is int
            and observed == generation, 'RESOURCE_NOT_RECONCILED')
    rows = [r for r in obj.get('status', {}).get('conditions', [])
            if type(r) is dict and r.get('type') == 'Ready']
    require(len(rows) == 1 and rows[0].get('status') == 'True', 'RESOURCE_NOT_READY')
    require(not obj.get('metadata', {}).get('deletionTimestamp'), 'RESOURCE_DELETING')


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


def routing_configuration(service):
    """Preserve behavioral annotations; omit only CLI diagnostic provenance."""
    ignored = {'run.googleapis.com/client-name', 'run.googleapis.com/client-version',
               'run.googleapis.com/operation-id', 'run.googleapis.com/urls',
               'serving.knative.dev/creator', 'serving.knative.dev/lastModifier'}
    def annotations(metadata):
        rows = metadata.get('annotations', {})
        require(type(rows) is dict, 'ANNOTATIONS_INVALID')
        return {k: v for k, v in rows.items() if k not in ignored}
    return (annotations(service.get('metadata', {})),
            annotations(service.get('spec', {}).get('template', {}).get('metadata', {})))


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


def pinned_traffic(service):
    ready(service)
    actual, requested = traffic(service, 'status'), traffic(service, 'spec')
    require(actual == requested, 'REQUESTED_EFFECTIVE_TRAFFIC_MISMATCH')
    return next(iter(actual))


def image_digest(revision):
    digest = revision.get('status', {}).get('imageDigest', '').rsplit('@', 1)[-1]
    require(re.fullmatch('sha256:[0-9a-f]{64}', digest or ''), 'IMAGE_DIGEST_INVALID')
    return digest


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
    for suffix in ('/ping', '/version'):
        try:
            req = urllib.request.Request(url + suffix, headers={'Cache-Control': 'no-cache'})
            with opener.open(req, timeout=20) as response:
                raw = response.read(262145)
                require(response.status == 200 and len(raw) <= 262144, 'HEALTH_HTTP_INVALID')
                result = json.loads(raw)
        except GuardError:
            raise
        except Exception:
            raise GuardError('HEALTH_REQUEST_FAILED') from None
        require(type(result) is dict and result.get('ok') is True, 'HEALTH_NOT_OK')
        if suffix == '/version':
            require(result.get('commit_sha') == commit and result.get('revision') == revision,
                    'HEALTH_IDENTITY_MISMATCH')
            if usage_required:
                u=result.get('usage_v6') or {}
                require(u.get('version')==USAGE_VERSION and u.get('mode')=='required' and u.get('ready') is True and u.get('database')=='postgresql',
                        'USAGE_NATIVE_HEALTH_NOT_READY')


def validate_candidate(before_spec, service, revision, name, commit, digest, usage_required=False):
    ready(revision)
    require(revision.get('metadata', {}).get('name') == name, 'CANDIDATE_NAME_MISMATCH')
    for spec in (template(service), revision.get('spec', {})):
        required_configuration(spec, commit)
        if usage_required:usage_configuration(spec)
        require(unaffected_spec(spec,usage_required) == unaffected_spec(before_spec,usage_required), 'UNRELATED_CONFIG_CHANGED')
        require(spec['containers'][0].get('image') == IMAGE_ROOT + '@' + digest, 'CANDIDATE_IMAGE_NOT_PINNED')
    require(image_digest(revision) == digest, 'CANDIDATE_DIGEST_MISMATCH')


def execute(project, commit, build, operation, expected_current, *, usage_required=False):
    require(project == PROJECT, 'WRONG_PROJECT')
    require(type(commit) is str and re.fullmatch('[0-9a-f]{40}', commit), 'COMMIT_SHA_INVALID')
    require(type(expected_current) is str and re.fullmatch('[0-9a-f]{40}', expected_current),
            'EXPECTED_CURRENT_SHA_INVALID')
    require(type(build) is str and re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', build),
            'BUILD_ID_INVALID')
    require(operation in {'deploy', 'rollback'}, 'OPERATION_INVALID')
    require(not usage_required or operation=='deploy', 'LEGACY_ROLLBACK_REQUIRES_MAINTENANCE')
    phase('validate_current_required')
    before = resource('services', SERVICE)
    prior = pinned_traffic(before)
    prior_revision = resource('revisions', prior)
    ready(prior_revision)
    required_configuration(prior_revision.get('spec', {}), expected_current)
    # A prior no-traffic stage, or a traffic-only rollback, can leave a newer
    # service template. The ACTIVE revision is the runtime identity. Permit
    # only image/COMMIT differences; all other settings remain identical.
    required_configuration(template(before))
    require(unaffected_spec(template(before),usage_required) == unaffected_spec(prior_revision.get('spec', {}),usage_required),
            'SERVICE_TEMPLATE_DIFFERS_FROM_ACTIVE')
    if expected_current == BASELINE_COMMIT:
        require(prior == BASELINE_REVISION and image_digest(prior_revision) == BASELINE_DIGEST,
                'BASELINE_IDENTITY_MISMATCH')
    names_before = revision_names()
    routing_before = routing_configuration(before)
    require(prior in names_before, 'PRIOR_REVISION_MISSING')
    if operation == 'rollback':
        phase('validate_pinned_required_rollback')
        target = resource('revisions', BASELINE_REVISION)
        ready(target)
        required_configuration(target.get('spec', {}), BASELINE_COMMIT)
        require(image_digest(target) == BASELINE_DIGEST, 'ROLLBACK_DIGEST_MISMATCH')
        require(BASELINE_REVISION in names_before, 'ROLLBACK_TARGET_MISSING')
        latest = resource('services', SERVICE)
        require(pinned_traffic(latest) == prior and revision_names() == names_before,
                'CONCURRENT_CHANGE_BEFORE_ROLLBACK')
        require(unaffected_spec(template(latest)) == unaffected_spec(template(before))
                and routing_configuration(latest) == routing_before,
                'CONCURRENT_CONFIG_CHANGE_BEFORE_ROLLBACK')
        phase('rollback_required_traffic_only')
        gcloud('run', 'services', 'update-traffic', SERVICE, '--project=' + PROJECT,
               '--region=' + REGION, '--to-revisions=' + BASELINE_REVISION + '=100', '--quiet')
        final = resource('services', SERVICE)
        require(routing_configuration(final) == routing_before, 'UNRELATED_ANNOTATIONS_CHANGED')
        require(pinned_traffic(final) == BASELINE_REVISION, 'ROLLBACK_TRAFFIC_MISMATCH')
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
           *(['--update-secrets=MM_USAGE_AUTHORITY_SECRET='+USAGE_SECRET['name']+':'+USAGE_SECRET['key']] if usage_required else []), '--quiet')
    staged = resource('services', SERVICE)
    candidate = resource('revisions', name)
    require(pinned_traffic(staged) == prior, 'TRAFFIC_CHANGED_BEFORE_VALIDATION')
    require(revision_names() == names_before | {name}, 'CONCURRENT_OR_UNEXPECTED_REVISION')
    require(routing_configuration(staged) == routing_before, 'UNRELATED_ANNOTATIONS_CHANGED')
    validate_candidate(template(before), staged, candidate, name, commit, digest, usage_required)
    tags = [r for r in staged.get('status', {}).get('traffic', [])
            if r.get('tag') == 'p6-candidate' and r.get('revisionName') == name]
    require(len(tags) == 1, 'CANDIDATE_TAG_MISSING')
    phase('check_candidate_identity_before_promotion')
    health(tags[0].get('url'), commit, name, **({'usage_required':True} if usage_required else {}))
    latest = resource('services', SERVICE)
    require(pinned_traffic(latest) == prior and revision_names() == names_before | {name},
            'CONCURRENT_CHANGE_BEFORE_PROMOTION')
    require(routing_configuration(latest) == routing_before, 'UNRELATED_ANNOTATIONS_CHANGED')
    validate_candidate(template(before), latest, resource('revisions', name), name, commit, digest, usage_required)
    phase('promote_exact_revision')
    gcloud('run', 'services', 'update-traffic', SERVICE, '--project=' + PROJECT,
           '--region=' + REGION, '--to-revisions=' + name + '=100', '--quiet')
    final = resource('services', SERVICE)
    require(pinned_traffic(final) == name, 'PROMOTION_TRAFFIC_MISMATCH')
    require(revision_names() == names_before | {name}, 'CONCURRENT_CHANGE_AFTER_PROMOTION')
    require(routing_configuration(final) == routing_before, 'UNRELATED_ANNOTATIONS_CHANGED')
    validate_candidate(template(before), final, resource('revisions', name), name, commit, digest, usage_required)
    health(final.get('status', {}).get('url'), commit, name, **({'usage_required':True} if usage_required else {}))
    return {'status': 'PASS_REQUIRED_DEPLOY', 'revision': name, 'previous_revision': prior,
            'runtime_commit_sha': commit, 'image_digest': digest, 'authority_mode': 'required',
            'new_revision': True, 'health_before_promotion': True, 'unrelated_spec_preserved': True,
            'usage_required':usage_required,'usage_native_health_verified':usage_required}


def main():
    try:
        result = execute(os.environ.get('MM_CB_PROJECT'), os.environ.get('MM_CB_COMMIT'),
                         os.environ.get('MM_CB_BUILD'), os.environ.get('MM_P6_OPERATION'),
                         os.environ.get('MM_P6_EXPECTED_CURRENT_SHA'), usage_required=True)
        print(json.dumps({**result, 'ask_calls': 0, 'provider_calls': 0,
                          'automatic_rollback': False, 'guard_version': 'phase6-integrated-usage-v2'}, sort_keys=True))
    except GuardError as exc:
        print(json.dumps({'status': 'FAIL_REQUIRED_RELEASE', 'phase': PHASE,
                          'code': str(exc), 'automatic_rollback': False}))
        raise SystemExit(2)
    except Exception:
        print(json.dumps({'status': 'FAIL_REQUIRED_RELEASE', 'phase': PHASE,
                          'code': 'UNEXPECTED_ERROR', 'automatic_rollback': False}))
        raise SystemExit(2)


if __name__ == '__main__':
    main()
