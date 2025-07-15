import pytest
from app.scraper import get_match_result

@pytest.mark.parametrize("score, expected_result", [
    ("1 - 0", "1"),
    ("0 - 1", "2"),
    ("2 - 2", "Χ"),
    (" 3 - 1 ", "1"), # Test with whitespace
    ("0 - 0", "Χ"),
])
def test_get_match_result_valid(score, expected_result):
    """Tests valid score strings."""
    assert get_match_result(score) == expected_result

@pytest.mark.parametrize("score, expected_result", [
    ("1-0", None),          # Invalid format (no spaces)
    ("a - b", None),        # Non-integer values
    ("1 - 2 - 3", None),    # Too many parts
    ("", None),             # Empty string
    (None, None),           # None input
    ("FT", None)            # Common non-score text
])
def test_get_match_result_invalid(score, expected_result):
    """Tests invalid or malformed score strings."""
    assert get_match_result(score) == expected_result 