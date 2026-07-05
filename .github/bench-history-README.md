# apex-14 benchmark history

Machine-collected criterion timings, one JSON snapshot per push to `main`, under
`results/<YYYY-MM-DD>-<short-sha>.json`. This branch is written automatically by
the `bench-history` GitHub Actions workflow; do not edit it by hand.

## Snapshot format

```json
{
  "git_sha": "abc1234",
  "timestamp": "2026-07-05T12:00:00Z",
  "results": [
    { "id": "dynamics_rhs/point_mass_4", "mean_ns": 3.51, "unit": "ns" }
  ]
}
```

- `id` — criterion `group/benchmark` name.
- `mean_ns` — criterion mean estimate, in the reported `unit`.

## Caveats

These are **GitHub-hosted-runner** numbers: shared CPUs, variable throttling, and
noisy neighbours. Treat them as **trend data** (has a hot path drifted over many
commits?), **not** absolute performance numbers, and **not** a pass/fail gate.
There is intentionally no regression gating wired to this data.
