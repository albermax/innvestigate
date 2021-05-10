from . import analyzer
from .analyzer import NotAnalyzeableModelException, create_analyzer

# Disable pyflaks warnings:
assert analyzer
assert create_analyzer
assert NotAnalyzeableModelException

__version__ = "1.0.9"
