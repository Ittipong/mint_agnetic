"""Test LLM response quality checks."""

import pytest
from tests.evaluation.evaluators import detect_shame, extract_numbers_from_response


class TestShameDetection:
    """Tests for shame/judgment language detection."""

    def test_no_shame_positive(self):
        """Positive responses should not trigger shame detection."""
        text = "เดือนนี้ใช้ไป 32,400 บาท — เหลืออีก 17,600 งบยังโอเคเลย"
        assert not detect_shame(text)

    def test_no_shame_neutral(self):
        """Neutral observation should not trigger shame detection."""
        text = "สังเกตว่าหมวดอาหารใช้ไปเยอะกว่าปกติ"
        assert not detect_shame(text)

    def test_shame_spending_too_much(self):
        """Should detect 'ใช้เยอะเกินไป'."""
        text = "ใช้เยอะเกินไปนะ ควรลดค่าใช้จ่าย"
        assert detect_shame(text)

    def test_shame_should_not(self):
        """Should detect 'ควรลด'."""
        text = "ควรลดค่าใช้จ่ายบางอย่าง"
        assert detect_shame(text)

    def test_shame_waste(self):
        """Should detect wasteful spending language."""
        text = "ไม่ควรซื้อของแพงขนาดนั้น"
        assert detect_shame(text)

    def test_shame_empty(self):
        """Empty text should not trigger shame."""
        assert not detect_shame("")
        assert not detect_shame(None)


class TestNumberExtraction:
    """Tests for number extraction from responses."""

    def test_has_thai_currency(self):
        """Thai currency format detected."""
        text = "ใช้ไป 32,400 บาท"
        assert extract_numbers_from_response(text)

    def test_has_percentage(self):
        """Percentage format detected."""
        text = "ใช้ไป 64.8%"
        assert extract_numbers_from_response(text)

    def test_has_days(self):
        """Days format detected."""
        text = "เหลืออีก 12 วัน"
        assert extract_numbers_from_response(text)

    def test_no_numbers(self):
        """Plain text without numbers."""
        text = "ยังไม่มีข้อมูลเดือนนี้เลยนะ"
        assert not extract_numbers_from_response(text)

    def test_empty(self):
        """Empty text returns False."""
        assert not extract_numbers_from_response("")
        assert not extract_numbers_from_response(None)
