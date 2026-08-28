# Contributing

This is research software. Changes must preserve the measurement definitions,
evidence chain, and reproducible configuration behavior documented in
`REPRODUCING.md` and the method-specific documentation.

1. Create a focused branch and describe the scientific or operational reason
   for the change.
2. Add or update hardware-free tests for every behavioral or numerical change.
3. Run `python -m pytest -q` from the repository root.
4. For D405 numerical changes, run synthetic 0%, threshold-level, and mid-fill
   checks. For camera changes, complete the manual smoke-test checklist.
5. Document changes to geometry, models, filters, calibration, thresholds,
   data schemas, or retained evidence in `CHANGELOG.md`.

Do not commit camera serial numbers, usernames, local absolute paths, model
weights, calibration arrays, raw recordings, credentials, or participant data.
Use an ignored `config.local.yaml` beside the tracked example configuration.

Pull requests that change an estimator must explain the expected scientific
effect. A synthetic pass alone is not evidence of accuracy or validation.
