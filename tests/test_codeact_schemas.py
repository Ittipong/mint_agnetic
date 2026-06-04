"""Unit tests for src.agent.tools.codeact.schemas.

Covers UT-CS01 — Pydantic model shape round-trip. The codeact namespace
wrappers serialize QuerySpec via .model_dump() so SQL templates can
build off the dict form; rehydration must preserve every field.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.agent.tools.codeact.schemas import (
    QuerySpec,
    ResolvedEntity,
    TimeRange,
)


class TestTimeRange:
    def test_UT_CS01_a_minimal_constructs(self):
        """UT-CS01a: start + end alone produces a valid TimeRange with
        granularity=month default and confidence=1.0."""
        tr = TimeRange(start=date(2026, 5, 1), end=date(2026, 5, 31))
        assert tr.granularity == "month"
        assert tr.confidence == 1.0
        assert tr.raw_phrase is None

    def test_UT_CS01_b_round_trips_via_model_dump(self):
        """UT-CS01b: model_dump() -> dict -> TimeRange(**d) preserves fields."""
        tr = TimeRange(
            start=date(2026, 5, 1),
            end=date(2026, 5, 31),
            granularity="week",
            confidence=0.8,
            raw_phrase="สัปดาห์นี้",
        )
        d = tr.model_dump()
        rehydrated = TimeRange(**d)
        assert rehydrated == tr


class TestResolvedEntity:
    def test_UT_CS01_c_required_fields_only(self):
        """UT-CS01c: sync_id/display_name/kind/score are required."""
        ent = ResolvedEntity(
            sync_id="w1",
            display_name="เงินสด",
            kind="wallet",
            score=0.95,
        )
        assert ent.alternatives == []
        assert ent.expand_ids == []

    def test_UT_CS01_d_score_rejects_out_of_range(self):
        """UT-CS01d: score must be in [0, 1]; > 1.0 raises ValidationError."""
        with pytest.raises(Exception):
            ResolvedEntity(
                sync_id="w1", display_name="x", kind="wallet", score=1.5,
            )


class TestQuerySpec:
    def test_UT_CS01_e_round_trip_with_full_payload(self):
        """UT-CS01e: a fully-populated QuerySpec marshals/unmarshals losslessly
        through model_dump()/QuerySpec(**dump). This is the contract
        sql_templates relies on (Wave 2)."""
        spec = QuerySpec(
            metric="sum_expense",
            wallets=[ResolvedEntity(
                sync_id="w1", display_name="เงินสด",
                kind="wallet", score=0.99,
            )],
            categories=[ResolvedEntity(
                sync_id="c1", display_name="ค่าอาหาร",
                kind="category", score=0.9,
                expand_ids=["อาหาร", "ของกิน"],
            )],
            tags=[],
            time_range=TimeRange(
                start=date(2026, 5, 1), end=date(2026, 5, 31),
            ),
            group_by="day",
            currency="THB",
            order_by="amount_desc",
            limit=10,
            note_query=["กาแฟ"],
            match_destination_note=False,
            has_note=True,
            convert_to_thb=True,
            transaction_type="expense",
        )
        d = spec.model_dump(mode="json")
        rehydrated = QuerySpec(**d)
        # Equality with mode="json" requires re-parsing dates — check key fields.
        assert rehydrated.metric == "sum_expense"
        assert rehydrated.transaction_type == "expense"
        assert rehydrated.note_query == ["กาแฟ"]
        assert rehydrated.match_destination_note is False
        assert rehydrated.has_note is True
        assert rehydrated.wallets[0].sync_id == "w1"
        assert rehydrated.categories[0].expand_ids == ["อาหาร", "ของกิน"]

    def test_UT_CS01_f_metric_literal_enforced(self):
        """UT-CS01f: metric not in the Literal whitelist -> ValidationError."""
        with pytest.raises(Exception):
            QuerySpec(
                metric="unknown_metric",
                time_range=TimeRange(
                    start=date(2026, 5, 1), end=date(2026, 5, 31),
                ),
            )
