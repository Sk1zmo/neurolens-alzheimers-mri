-- =====================================================================
-- NeuroLens — Supabase schema
-- Run once in the Supabase SQL editor (Dashboard -> SQL -> New query).
-- =====================================================================
--
-- Every uploaded scan is written twice, on purpose, for two different
-- readers:
--
--   public.scans           the USER's copy   -> rendered in /history
--   public.training_queue  the MODEL's copy  -> consumed by training/retrain.py
--
-- They are separate rows rather than one row with flags because their
-- lifecycles genuinely differ: a user deleting their history must not silently
-- shrink the training corpus, and a reviewer re-labelling a scan for training
-- must not rewrite what the user was originally shown.
--
-- Access model: there are no end-user accounts. RLS is ON with no permissive
-- policy, so the anon/public key can read nothing at all. Every read and write
-- goes through the Next.js route handlers using the service-role key, which
-- scope queries by the caller's session id. Keep SUPABASE_SERVICE_ROLE_KEY
-- server-side only — it must never be exposed with a NEXT_PUBLIC_ prefix.

create extension if not exists "pgcrypto";

-- ---------------------------------------------------------------- scans ---
create table if not exists public.scans (
  id                  uuid primary key default gen_random_uuid(),
  session_id          text        not null,
  storage_path        text        not null,
  original_filename   text,
  mime_type           text,
  byte_size           integer,
  width               integer,
  height              integer,

  -- model output at the time of upload (a snapshot; never rewritten by a
  -- later retrain, so history stays a faithful record of what was shown)
  predicted_class_id  smallint    not null,
  predicted_label     text        not null,
  confidence          real        not null,
  probabilities       real[]      not null,
  margin              real,
  energy              real,
  out_of_distribution boolean     not null default false,
  input_check         jsonb,

  model_name          text,
  model_version       text,
  note                text,
  created_at          timestamptz not null default now()
);

create index if not exists scans_session_created_idx
  on public.scans (session_id, created_at desc);
create index if not exists scans_created_idx
  on public.scans (created_at desc);

-- -------------------------------------------------------- training_queue ---
create table if not exists public.training_queue (
  id                   uuid primary key default gen_random_uuid(),
  scan_id              uuid references public.scans (id) on delete set null,
  session_id           text        not null,
  storage_path         text        not null,

  -- The model's own guess. Recorded for triage only. retrain.py never treats
  -- this as ground truth: training on your own predictions just amplifies
  -- whatever the model already believes.
  predicted_label      text        not null,
  predicted_confidence real,

  -- Ground truth, supplied by a human in /review. Null means "not usable yet".
  verified_label       text        check (
    verified_label is null or verified_label in (
      'NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented'
    )
  ),
  reviewed_by          text,
  reviewed_at          timestamptz,
  review_note          text,

  status               text        not null default 'pending'
                       check (status in ('pending', 'approved', 'rejected')),
  used_in_training     boolean     not null default false,
  consent              boolean     not null default true,
  created_at           timestamptz not null default now()
);

create index if not exists training_queue_pending_idx
  on public.training_queue (status, used_in_training, created_at);
create index if not exists training_queue_ready_idx
  on public.training_queue (used_in_training, status)
  where verified_label is not null;

-- ------------------------------------------------------------------ RLS ---
alter table public.scans          enable row level security;
alter table public.training_queue enable row level security;

-- No policies are created on purpose. With RLS enabled and zero permissive
-- policies, the anon key is denied everything; the service-role key bypasses
-- RLS and is only ever used from the server. If you later add real user
-- accounts, add auth.uid()-scoped policies here and drop the service-role
-- reads from the route handlers.

-- --------------------------------------------------------------- storage ---
-- Create the bucket (private). Safe to re-run.
insert into storage.buckets (id, name, public)
values ('scans', 'scans', false)
on conflict (id) do nothing;

-- ----------------------------------------------------------------- views ---
-- Aggregates for the /performance dashboard. Exposed through the server-side
-- route handler only.
create or replace view public.scan_stats as
select
  count(*)                                          as total_scans,
  count(distinct session_id)                        as total_sessions,
  count(*) filter (where out_of_distribution)       as flagged_out_of_distribution,
  avg(confidence)                                   as mean_confidence,
  max(created_at)                                   as last_upload_at
from public.scans;

create or replace view public.retraining_stats as
select
  count(*)                                                       as queued_total,
  count(*) filter (where verified_label is not null)             as labelled,
  count(*) filter (where verified_label is not null
                     and status = 'approved'
                     and not used_in_training)                   as ready_for_training,
  count(*) filter (where used_in_training)                       as already_trained,
  count(*) filter (where status = 'rejected')                    as rejected
from public.training_queue;
