# Contributing

Thank you for your interest in InferDrift. It is an early preview, so please
open an issue to discuss a larger change before you start work on it.

## Development setup

```bash
git clone https://github.com/ChanYCTobias/InferDrift.git
cd InferDrift
python -m pip install -e .
python -m pip install pytest
python -m pytest
```

The tests run offline against the local stand-in endpoint and need no model,
GPU or API key. Please keep it that way: a test must not need network access.

## Pull requests

- Keep each pull request focused on one change, and add or update tests for it.
- Keep the core package free of third-party dependencies. Code that needs
  PyTorch or transformers belongs behind the `hf` extra.
- Never put credentials, tokens or private data in code, tests or fixtures.
- Describe the change, and how you tested it, in the pull request.

By contributing, you agree that your contributions are licensed under the
Apache License 2.0, as stated in section 5 of the [LICENSE](LICENSE).

## Reporting security problems

Do not report security problems in public issues; see [SECURITY.md](SECURITY.md).
