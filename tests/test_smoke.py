"""Phase-1 smoke tests: package imports and CLI metadata work."""

from typer.testing import CliRunner

from tedsds import __version__
from tedsds.cli import app

runner = CliRunner()


def test_version_matches_package() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help_shows_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ping" in result.stdout
    assert "version" in result.stdout
