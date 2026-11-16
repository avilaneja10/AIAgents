# Weather codes (METAR style)

| code | meaning |
|---|---|
| `CLR` | clear |
| `-RA` | light rain |
| `RA` | moderate rain |
| `+RA` | heavy rain |
| `BR` | mist |
| `FG` | fog |
| `SN` | snow |
| `TS` | thunderstorm |

`visibility_mi` is visibility in statute miles.

## Bad weather
An airport-hour counts as **bad weather** if and only if:
- `wx_code` is `TS`, `SN` or `+RA`; **or**
- `wx_code` is `FG` **and** `visibility_mi` < 1.0.

Everything else (`CLR`, `-RA`, `RA`, `BR`, and `FG` with visibility >= 1.0) is **not** bad weather.
