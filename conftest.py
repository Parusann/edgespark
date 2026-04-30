"""Pytest configuration.

Its mere presence at the repository root makes pytest prepend the root to
``sys.path``, so ``from edgespark import ...`` resolves in a fresh clone without
installing the package or setting ``PYTHONPATH``. Keeps the quickstart honest:
``pip install numpy pyyaml pytest && pytest`` just works.
"""
