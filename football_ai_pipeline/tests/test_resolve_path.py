"""Tests for ui.app.resolve_path helper."""

from pathlib import Path
import sys

# Ensure the ui module is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui.app import resolve_path


def test_none_returns_none():
    assert resolve_path(None, Path("/base")) is None


def test_empty_string_returns_none():
    assert resolve_path("", Path("/base")) is None
    assert resolve_path("   ", Path("/base")) is None


def test_absolute_path_returned_as_is():
    if sys.platform == "win32":
        result = resolve_path("C:/foo/bar.yaml", Path("D:/base"))
        assert str(result) == r"C:\foo\bar.yaml"
    else:
        result = resolve_path("/foo/bar.yaml", Path("/base"))
        assert result == Path("/foo/bar.yaml")


def test_relative_path_joined_to_base(tmp_path):
    result = resolve_path("configs/default.yaml", tmp_path)
    assert result == (tmp_path / "configs" / "default.yaml").resolve()


def test_duplicate_segment_stripped(tmp_path):
    """If relative path starts with base_dir's folder name, strip it."""
    base = tmp_path / "football_ai_pipeline"
    base.mkdir()
    result = resolve_path("football_ai_pipeline/configs/default.yaml", base)
    expected = (base / "configs" / "default.yaml").resolve()
    assert result == expected


def test_no_false_strip(tmp_path):
    """Don't strip if first segment doesn't match base_dir name."""
    base = tmp_path / "football_ai_pipeline"
    base.mkdir()
    result = resolve_path("configs/default.yaml", base)
    expected = (base / "configs" / "default.yaml").resolve()
    assert result == expected


def test_single_duplicate_segment(tmp_path):
    """Edge case: relative path is just the base_dir name itself."""
    base = tmp_path / "myproject"
    base.mkdir()
    result = resolve_path("myproject", base)
    assert result == base.resolve()
