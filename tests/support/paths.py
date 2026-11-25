"""Where the tests find the repository's files"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] # the checkout (or the copy tests/run_wheel_tests.py makes)
VECTORS = ROOT/"tests"/"vectors" # reference vectors: what each is, in its "source"
