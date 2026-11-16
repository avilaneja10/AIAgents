# Flight operations dataset (January-February 2025)

## Files
| file | contents |
|---|---|
| `airlines/<CARRIER>.csv` | one file per carrier: DL (Delta), AA (American), UA (United), WN (Southwest), B6 (JetBlue), AS (Alaska) |
| `weather.csv` | hourly weather observation at each airport |
| `airports.csv` | airport metadata: `airport, city, timezone, utc_offset_hours` |
| `weather_codes.md` | meaning of weather codes and the definition of **bad weather** |

### Flights columns
`flight_id, date, origin, dest, sched_dep, actual_dep, cancelled`
- `flight_id` is unique: every flight appears exactly once. The export may contain repeated records.
- `date` is the scheduled departure date; `sched_dep` / `actual_dep` are clock times `HH:MM`.
- A delayed flight may depart after midnight, i.e. on the day after `date`.
- `actual_dep` is empty if the flight was cancelled. `cancelled` is `true` / `false`.
- Times are local time at the origin airport, **unless the column name ends in `_utc`**.

### Weather columns
`airport, date, hour, wx_code, visibility_mi` -- local time at that airport; `hour` h covers h:00-h:59.

## Conventions
- Departure delay (minutes) = actual departure - scheduled departure. Early departures are negative.
- A flight is **delayed** if its departure delay is **more than 15 minutes**.
- **Delay rate** = delayed flights / completed (non-cancelled) flights. Cancelled flights are excluded
  from both numerator and denominator. Report rates as percentages with 1 decimal.
- Dates, months and hours refer to local time at the origin airport.
