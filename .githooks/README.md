# Git hooks

## Pre-push hook

The `pre-push` hook runs the **eye_test_engine** unit tests before allowing `git push`. If any test fails, the push is aborted.

### One-time setup

From the repo root, run:

```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-push
```

After this, every `git push` will run the tests first.

### Skip hook once (not recommended)

To push without running tests in an emergency:

```bash
git push --no-verify
```

### Restore default hooks

To stop using these hooks:

```bash
git config --unset core.hooksPath
```
