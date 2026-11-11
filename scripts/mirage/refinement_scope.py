"""Reject empty numerical scopes before any model work or success reporting."""


def numerical_names(requested, *, geometry_only=False, numerical_only=False):
    base=[name for name in requested if not name.startswith('holdout_')]
    if not geometry_only and not base:
        raise ValueError('zero numerical cases; use explicit geometry-only scope')
    if numerical_only and len(base)!=len(requested):
        raise ValueError('numerical phase requires only frozen base fixtures')
    return base
