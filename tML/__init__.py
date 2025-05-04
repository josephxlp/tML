"""Top-level package for tML."""

__author__ = """J"""
__email__ = "l.joseph.p@hotmail.com"
__version__ = "0.0.1"

from .pagluon import agluon_pipeline
from .pamljar import amljar_pipeline

__all__ = [
    "agluon_pipeline",
    "amljar_pipeline"
]
