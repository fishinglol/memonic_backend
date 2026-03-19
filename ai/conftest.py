"""
conftest.py — Ensure heavy optional modules are mocked before test collection.

The ai/api.py imports 'memory' which depends on mem0/ChromaDB.
We mock it at sys.modules level so pytest can collect test_enroll.py safely.
"""
import sys
from unittest.mock import MagicMock

# Mock the memory module so api.py can be imported without mem0/ChromaDB
if "memory" not in sys.modules:
    mock_memory = MagicMock()
    mock_memory.init_memory = MagicMock()
    mock_memory.silence_watcher = MagicMock()
    sys.modules["memory"] = mock_memory
