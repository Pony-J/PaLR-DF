# Data card

**Provider:** Center for Orbit Determination in Europe (CODE), AIUB, distributed through IGS product infrastructure. **Product:** `COD0MGXFIN`, RINEX clock 3.04, daily 30 s satellite clocks. The original product header provides [DOI 10.48350/185742](https://doi.org/10.48350/185742). Data terms are linked in [DATA_LICENSE.md](../DATA_LICENSE.md).

The release includes only the columns and source days needed for the first paper's core experiments. The 27 source days are Jan 1–10, Jan 12–14, Jan 16–23, and Jan 25–30, 2024. The raw-product manifest records source filenames, original IGN mirror locations, compressed and uncompressed SHA-256 values. Dates omitted here are not required by the released interval recipes.

Satellites: `C06 C09 C10 C11 C14 C19 C20 C21 C22 C24 C25 C26 C27 C28 C29 C30 C32 C33 C34 C35 C36 C37 C39 C41 C42 C44 C45 C46`.

| NPZ field | Shape / type | Meaning |
|---|---|---|
| `epochs` | 77,760, `datetime64[ns]` | GPS time labels from RINEX records; no UTC conversion |
| `satellites` | 28 strings | Column order |
| `bias_ns` | 77,760 × 28, float64 | RINEX AS bias in seconds multiplied by 1e9 |
| `time_system` | scalar string | `GPS` |
| `sampling_seconds` | scalar integer | 30 |

Each source day is validated for 2,880 unique ordered epochs, midnight origin, exact 30 s spacing and finite biases. Conversion performs no interpolation, smoothing, imputation or clock alignment. Gaps between packaged date intervals are intentional; the loader requires complete contiguous raw ranges for each configured target interval.

To regenerate the asset, obtain the manifest-listed products from a public IGS/CODE archive, decompress them without changing contents, then run:

```bash
palr-df prepare-data --raw-dir /path/to/CODE_CLK --output runs/rebuilt_clock.npz
```

The command checks every uncompressed product hash before parsing. [IGS data access](https://igs.org/data/) lists archive access routes; CODE's [official download service](https://www.aiub.unibe.ch/download) is also available. Historical mirror locations are provenance records and can change with archive operations. The distributed processed asset makes normal replay independent of archive availability.
