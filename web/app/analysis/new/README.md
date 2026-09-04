# New Analysis

Authenticated customer flow for creating a CourtFlux analysis.

- Supports direct private video upload through Supabase Storage or a YouTube URL.
- Requires only home and away team names; date, competition, kit colours, roster and jersey recognition remain optional.
- Estimates analysis-minute usage from the selected time window before submission.
- Prevents starting a run when the selected window exceeds the remaining monthly allowance.
- Queues the existing CourtFlux worker pipeline through `/api/jobs`.
