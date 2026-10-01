-- MachineMind Phase 6. Explicit, additive migration; never run at import/request time.
-- Execute against the SAME database/user used by mm-ai-ingest-prod, during cutover.
-- No document, company or existing usage table is deleted or rewritten.
BEGIN;
SET LOCAL lock_timeout = '5s';
CREATE TABLE IF NOT EXISTS public.mm_ai_usage_owner_v6 (
    company_id text PRIMARY KEY CHECK (length(company_id) BETWEEN 1 AND 128)
);
CREATE TABLE IF NOT EXISTS public.mm_ai_usage_bucket_v6 (
    company_id text NOT NULL REFERENCES public.mm_ai_usage_owner_v6(company_id),
    period_kind text NOT NULL CHECK (period_kind IN ('day','month')),
    period_key text NOT NULL,
    opening_count bigint NOT NULL CHECK (opening_count >= 0),
    admitted_count bigint NOT NULL DEFAULT 0 CHECK (admitted_count >= 0),
    PRIMARY KEY(company_id,period_kind,period_key)
);
CREATE TABLE IF NOT EXISTS public.mm_ai_usage_request_v6 (
    company_id text NOT NULL REFERENCES public.mm_ai_usage_owner_v6(company_id),
    request_id text NOT NULL,
    actor_id text NOT NULL,
    operation text NOT NULL,
    payload_sha256 text NOT NULL CHECK (length(payload_sha256)=64),
    day_key text NOT NULL,
    month_key text NOT NULL,
    state text NOT NULL CHECK (state IN ('started','settled','uncertain','rejected')),
    counted boolean NOT NULL DEFAULT true,
    cap_usd numeric(18,8) NOT NULL CHECK (cap_usd >= 0),
    known_usd numeric(18,8) NOT NULL DEFAULT 0 CHECK (known_usd >= 0),
    uncertain_usd numeric(18,8) NOT NULL CHECK (uncertain_usd >= 0),
    result_code text NOT NULL DEFAULT 'STARTED',
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    completed_at timestamptz,
    PRIMARY KEY(company_id,request_id)
);
CREATE INDEX IF NOT EXISTS mm_ai_usage_request_v6_pending
 ON public.mm_ai_usage_request_v6 (company_id,created_at) WHERE state IN ('started','uncertain');
CREATE TABLE IF NOT EXISTS public.mm_ai_usage_schema_v6 (
    singleton boolean PRIMARY KEY DEFAULT true CHECK(singleton),
    version text NOT NULL CHECK(version='p6-interactive-ledger-v1')
);
INSERT INTO public.mm_ai_usage_schema_v6(singleton,version)
 VALUES (true,'p6-interactive-ledger-v1') ON CONFLICT(singleton) DO NOTHING;
COMMIT;
