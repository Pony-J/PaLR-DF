# Contributing

Please open an issue before changing scientific behavior. Include a minimal example, dependency versions and the relevant report. Keep the historical frozen result records unchanged; a new result should identify its population, date intervals, seeds and config separately.

For code changes, run the CPU tests, `verify-results` and `figures` commands from the README. A model or loader change should also demonstrate agreement with the pretrained reference states and original window alignment. Record runtime and numerical tolerances for any device-specific differences.

Do not include large assets in commits. Release assets are versioned separately with SHA-256 manifests. Contributions are accepted under the repository's MIT license; third-party code and data must retain their own notices.
