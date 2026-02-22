"""Pytest configuration and fixtures for MLOps pipeline tests."""

from hypothesis import settings, Verbosity

# Register hypothesis profiles
settings.register_profile("dev", max_examples=10)
settings.register_profile("ci", max_examples=100, verbosity=Verbosity.verbose)

# Load dev profile by default for local development
settings.load_profile("dev")
