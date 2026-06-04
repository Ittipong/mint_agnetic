"""Seed the dev DB with the deterministic eval fixtures.

Per `docs/v3/phase1_eval_construction_plan.md` §1.2, the eval set needs two
fixtures beyond the existing dev seed:

  (a) `NO_WALLET_USER` — a separate test user with 0 wallets, to exercise
      the `wallet_required` CTA path in ADD-03.
  (b) Optional `งบอาหาร` budget for ANA-03 (budget-vs-actual) — only
      seeded if the row needs deterministic budget ground truth.

This script is IDEMPOTENT: re-runs use `INSERT … ON CONFLICT DO NOTHING`.
Run BEFORE `evals/build_dataset.py` so the SQL ground-truth validations
have rows to read.

CLI:
  python -m evals.seed_db                    # apply all seeds (idempotent)
  python -m evals.seed_db --dry-run          # print SQL, don't execute
  python -m evals.seed_db --no-wallet-only   # just the NO_WALLET_USER row

The DSN comes from BACKEND_DATABASE_URL (memory
`reference_dev_db_write_access` — write access via psycopg + the .env.dev
DSN; the MCP postgres-dev is READ-ONLY).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Lazy import — psycopg is heavy; only needed for live runs.


# Fixed UUIDs for the seeded test users so eval rows can reference them by
# constant. UUIDv4 in the `00…00` range so the dev DB indexes treat them as
# regular rows; no collision risk with the active seed user
# `ba91d8a5-46b2-46f7-aaf4-189a54e17fe9`.
NO_WALLET_USER = "00000000-0000-4000-a000-000000000099"


# ---------------------------------------------------------------------------
# SQL fragments — keep idempotent (ON CONFLICT DO NOTHING).
# ---------------------------------------------------------------------------

SQL_NO_WALLET_USER = """
INSERT INTO users (id, sync_id, email, display_name, created_at)
VALUES (
    %(uid)s::uuid,
    %(uid)s,
    'eval-no-wallet@mint.test',
    'Eval No-Wallet User',
    now()
)
ON CONFLICT (id) DO NOTHING;
"""

# Comment the DB so a future investigator knows where the seed version
# pinned in eval row metadata came from.
SQL_DB_COMMENT = "COMMENT ON DATABASE mint_money_dev IS %(comment)s;"

DEFAULT_DB_COMMENT = "eval seed v1 — 2026-05-28"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _resolve_dsn() -> str:
    dsn = os.getenv("BACKEND_DATABASE_URL")
    if not dsn:
        # Look in mint_agentic_v3/.env then backend/.env.dev (per
        # reference_dev_db_write_access — the backend env carries write DSN).
        candidates = [
            Path(__file__).resolve().parent.parent / ".env",
            Path(__file__).resolve().parents[2] / "backend" / ".env.dev",
        ]
        for env_path in candidates:
            if not env_path.exists():
                continue
            for line in env_path.read_text().splitlines():
                if line.startswith("BACKEND_DATABASE_URL="):
                    dsn = line.split("=", 1)[1].strip().strip('"').strip("'")
                    os.environ["BACKEND_DATABASE_URL"] = dsn
                    break
            if dsn:
                break
    if not dsn:
        raise SystemExit(
            "BACKEND_DATABASE_URL is unset; cannot seed. Export it (see "
            "memory reference_dev_db_write_access) or place it in "
            "mint_agentic_v3/.env."
        )
    return dsn


def apply_no_wallet_user(conn, *, dry_run: bool) -> None:
    """Insert the 0-wallet test user. Idempotent."""
    print(f"  [seed] NO_WALLET_USER id={NO_WALLET_USER}")
    if dry_run:
        print(f"  [dry-run] {SQL_NO_WALLET_USER.strip()}")
        return
    with conn.cursor() as cur:
        cur.execute(SQL_NO_WALLET_USER, {"uid": NO_WALLET_USER})


def apply_db_comment(conn, comment: str, *, dry_run: bool) -> None:
    """Tag the database with the seed-version sentinel."""
    print(f"  [seed] DB comment = {comment!r}")
    if dry_run:
        print(f"  [dry-run] {SQL_DB_COMMENT.strip()} -- {comment}")
        return
    with conn.cursor() as cur:
        cur.execute(SQL_DB_COMMENT, {"comment": comment})


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="seed dev DB for v3 eval gate")
    parser.add_argument("--dry-run", action="store_true",
                        help="print SQL without executing")
    parser.add_argument("--no-wallet-only", action="store_true",
                        help="only apply the NO_WALLET_USER seed")
    parser.add_argument("--comment", default=DEFAULT_DB_COMMENT,
                        help="DB comment sentinel (eval seed version)")
    args = parser.parse_args(argv)

    dsn = _resolve_dsn()
    print(f"[seed_db] target DSN: {dsn[:30]}… (host hidden)")
    print(f"[seed_db] dry_run={args.dry_run}")

    if args.dry_run:
        # Dry-run still echoes the SQL but does not need a connection.
        apply_no_wallet_user(conn=None, dry_run=True)
        if not args.no_wallet_only:
            apply_db_comment(conn=None, comment=args.comment, dry_run=True)
        return 0

    import psycopg  # noqa: WPS433 — local import (heavy dependency)

    with psycopg.connect(dsn, autocommit=False) as conn:
        apply_no_wallet_user(conn, dry_run=False)
        if not args.no_wallet_only:
            apply_db_comment(conn, args.comment, dry_run=False)
        conn.commit()
    print("[seed_db] OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
