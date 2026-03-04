Legacy test suites are intentionally excluded from the default pytest gate.
They are retained as reference coverage for deprecated/removed snapshot paths and may not pass without extra setup or restoration work.

Run only legacy tests:

```bash
pytest tests/legacy -m "legacy"
```

Run one legacy module:

```bash
pytest tests/legacy/test_snapshot_functional.py -m "legacy"
```
