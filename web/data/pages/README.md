# Editable public page content

CourtFlux public/static web copy lives in JSON so positioning, SEO copy, FAQs, CTAs and section text can be edited without rewriting page components.

## Files

- `home.json` — homepage copy, pricing, capability/status copy and footer links
- `basketball-video-analysis.json` — core SEO landing page
- `coaches.json` — individual-coach landing page
- `teams.json` — team-staff landing page
- `academies.json` — unpublished positioning placeholder; intentionally blocked on roadmap #16
- `trust.json` — trust/security copy
- `video-data.json` — source-video and derived-data retention copy
- `privacy.json` — private-beta privacy policy copy
- `terms.json` — private-beta terms copy
- `ai-limitations.json` — AI evidence-boundary and failure-mode copy
- `shared.json` — shared legal navigation/footer labels

## Editing rules

1. Keep JSON valid. `web/tsconfig.json` has `resolveJsonModule` enabled, so the Next.js build imports these files directly.
2. Keep URLs as application paths such as `/login?mode=signup` or `/trust` unless an external destination is intentionally required.
3. Do not claim roadmap capabilities as live. The customer-facing beta currently centres on player/team tracking, ball detection/tracking and processing diagnostics.
4. Keep the 60-day direct-upload source-video retention rule consistent across public pages. Derived analysis results/statistics remain after source deletion.
5. Do not publish `academies.json` as a route until the Academy architecture/product scope in roadmap #16 is ready.
6. Brand assets, component logic, accessibility markup, responsive layout and authenticated/product state stay in React/CSS. JSON is for editable public-page content, not application logic.

After editing JSON, run the normal Web quality workflow (typecheck, lint and build) before merging.
