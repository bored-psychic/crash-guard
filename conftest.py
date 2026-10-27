"""Put the repo root on sys.path so tests can `import config`, `pipeline.*`,
`ctol.*` no matter where pytest is invoked from."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
