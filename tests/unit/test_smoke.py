"""Smoke tests to verify test infrastructure works."""


class TestInfrastructure:
    """Verify test setup is working."""

    def test_pytest_runs(self) -> None:
        """Basic test to verify pytest executes."""
        assert True

    def test_can_import_node_fdm(self) -> None:
        """Verify main package is importable."""
        import node_fdm

        assert node_fdm is not None

    def test_fixtures_dir_exists(self, fixtures_dir) -> None:
        """Verify fixtures directory fixture works."""
        # Directory may not exist yet, just check fixture works
        assert fixtures_dir is not None
