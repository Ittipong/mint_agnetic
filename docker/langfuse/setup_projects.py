#!/usr/bin/env python3
"""Provision the SEPARATE Langfuse project for the insight service — reproducibly.

Self-hosted Langfuse (OSS) has no project-provisioning public API and no
organization API keys, but it DOES accept the admin credentials (which are
reproducible via the compose LANGFUSE_INIT_USER_* vars) over its internal
auth+tRPC endpoints. So after `docker compose up -d` recreates the org + the
`mint-chat` project (LANGFUSE_INIT), run this once to (re)create the second
project and wire its keys into the insight service — surviving a full `down -v`.

What it does (idempotent):
  1. sign in as the admin user (NextAuth credentials flow → session cookie);
  2. ensure a project named `--project` exists under the org (create if missing);
  3. mint an API key for it (only if the insight .env doesn't already hold one);
  4. write LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST into the
     insight service's .env so the insight runs land in THIS project.

Usage (no extra deps — stdlib only):
  python docker/langfuse/setup_projects.py \
      [--host http://localhost:3100] [--project mint-insight] \
      [--email admin@mint.local] [--password mintadmin123] \
      [--insight-env ../../mint_insight/.env]
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _opener():
    cj = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))


def _login(op, host: str, email: str, password: str) -> None:
    csrf = json.load(op.open(f"{host}/api/auth/csrf"))["csrfToken"]
    body = urllib.parse.urlencode({
        "csrfToken": csrf, "email": email, "password": password, "json": "true",
    }).encode()
    op.open(urllib.request.Request(
        f"{host}/api/auth/callback/credentials", data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"}))
    sess = json.load(op.open(f"{host}/api/auth/session"))
    if not (sess.get("user") or {}).get("email"):
        raise SystemExit("login failed — check admin email/password")


def _trpc(op, host: str, proc: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{host}/api/trpc/{proc}", data=json.dumps({"json": payload}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        raw = op.open(req).read().decode()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"tRPC {proc} failed: {e.code} {e.read()[:200].decode()}")
    return json.loads(raw)["result"]["data"]["json"]


def _find_project(op, host: str, name: str):
    """Return (org_id, project_id|None) for the project named `name`."""
    sess = json.load(op.open(f"{host}/api/auth/session"))
    for org in (sess.get("user") or {}).get("organizations", []):
        for proj in org.get("projects", []):
            if proj.get("name") == name:
                return org["id"], proj["id"]
        # remember the first org as the creation target
        _find_project._org = org["id"]  # type: ignore[attr-defined]
    return getattr(_find_project, "_org", None), None


def _write_env(env_path: Path, host: str, public: str, secret: str) -> None:
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    want = {"LANGFUSE_ENABLED": "on", "LANGFUSE_PUBLIC_KEY": public,
            "LANGFUSE_SECRET_KEY": secret, "LANGFUSE_HOST": host}
    seen = set()
    for i, ln in enumerate(lines):
        m = re.match(r"^(LANGFUSE_[A-Z_]+)=", ln)
        if m and m.group(1) in want:
            lines[i] = f"{m.group(1)}={want[m.group(1)]}"
            seen.add(m.group(1))
    if not seen:
        lines.append("\n# ── Langfuse (insight service — separate project) ──")
    for k, v in want.items():
        if k not in seen:
            lines.append(f"{k}={v}")
    env_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:3100")
    ap.add_argument("--project", default="mint-insight")
    ap.add_argument("--email", default="admin@mint.local")
    ap.add_argument("--password", default="mintadmin123")
    ap.add_argument("--insight-env",
                    default=str((here / "../../../mint_insight/.env").resolve()))
    args = ap.parse_args()

    op = _opener()
    _login(op, args.host, args.email, args.password)

    org_id, proj_id = _find_project(op, args.host, args.project)
    if proj_id:
        print(f"✓ project '{args.project}' already exists ({proj_id})")
    else:
        proj = _trpc(op, args.host, "projects.create",
                     {"name": args.project, "orgId": org_id})
        proj_id = proj["id"]
        print(f"+ created project '{args.project}' ({proj_id})")

    env_path = Path(args.insight_env)
    cur = env_path.read_text() if env_path.exists() else ""
    # Reuse an existing non-shared key if the insight .env already has one.
    m = re.search(r"^LANGFUSE_PUBLIC_KEY=(pk-lf-[\w-]+)", cur, re.M)
    if m and m.group(1) != "pk-lf-mint-poc-public":
        print(f"✓ insight .env already keyed ({m.group(1)[:16]}…) — leaving as is")
        return
    key = _trpc(op, args.host, "projectApiKeys.create",
                {"projectId": proj_id, "note": "insight-service (setup_projects.py)"})
    _write_env(env_path, args.host, key["publicKey"], key["secretKey"])
    print(f"+ minted key {key['publicKey'][:20]}… → wrote {env_path}")
    print("→ reload the insight service to pick up the new keys")


if __name__ == "__main__":
    main()
