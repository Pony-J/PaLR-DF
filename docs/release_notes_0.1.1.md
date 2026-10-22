# PaLR-DF v0.1.1

Source and presentation update aligned with the October 22, 2026 manuscript submission.

- Add a paper-to-artifact [review guide](https://github.com/Pony-J/PaLR-DF/blob/v0.1.1/docs/reviewer_guide.md).
- Use the original submitted overview diagram, extracted without redrawing.
- Regenerate all seven result figures from the frozen data, using the manuscript's Informer display name and single-panel Fig. 9.
- Retain original manuscript plot PNGs and add manuscript-name / satellite-quantile table exports.
- Fix handling of parent segments in Windows asset paths, with a regression check for containment.

The frozen numerical records, model definitions, training recipes and pretrained weights are unchanged. This source version uses the existing [PaLR-DF-v0.1.0-assets.zip](https://github.com/Pony-J/PaLR-DF/releases/download/v0.1.0/PaLR-DF-v0.1.0-assets.zip) (244,422,396 bytes; SHA-256 `7bbd3c3fa840a9db5524f3ba397f6ee39dd111e4c673acde2f4749984499e8fe`). Extract it into the source directory to obtain `artifacts/`.

This release supplies the updated standalone source archive, Python wheel and their SHA-256 checksums. The source archive includes the guides, frozen result records, model configs, original result plot PNGs and tests. Run commands from its extracted root; the wheel alone does not include the external data, configs or result records.

Validation for this update: 13 CPU tests passed; complete frozen aggregate verification passed; all seven plots regenerated and visually checked; anonymous downloads of all v0.1.0 release files matched their checksums; all 582 downloaded asset payloads verified; the selected 12 h / 24 h PaLR-DF CPU replays and the V2 spectral recomputation passed. The original full 560-state release validation remains in `results/validation.json`.
