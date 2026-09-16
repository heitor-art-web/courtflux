# Editable public page content

CourtFlux public/static web copy lives in JSON so positioning, SEO copy, FAQs, CTAs and section text can be edited without rewriting page components.

## Files

- `home.json` — homepage metadata, navigation, ordered section composition, pricing and footer links
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

## Homepage composition

`home.json` is the source of truth for both homepage copy and section order. The `sections` array is rendered in sequence by `web/app/home-sections.tsx`.

Supported homepage section types are currently:

- `hero`
- `valueStrip`
- `story`
- `evidence`
- `audiences`
- `how`
- `pricing`
- `faq`
- `finalCta`

Editors may reorder sections, change copy, links and media, choose supported `story.variant` values, or remove a section from the array without changing `page.tsx`. New visual behaviours still belong in React/CSS and should be added as explicit typed section variants rather than embedded as logic in JSON.

Keep the `pricing` section id as `pricing` while authenticated account links continue to target `/#pricing`.

## Languages

English is the editable source language for the public JSON files. CourtFlux also supports Portuguese for Portugal (`pt-PT`) and Spanish for Spain (`es-ES`). Public localized URLs use `/pt-pt/...` and `/es-es/...`; English remains unprefixed.

Translations are maintained in `web/lib/i18n/pt-PT.ts` and `web/lib/i18n/es-ES.ts`, with small scoped overlays allowed for major copy migrations such as `web/lib/i18n/homepage-v2.ts`. Public pages are localized on the server so localized metadata, canonical URLs, hreflang entries, structured data and visible copy stay aligned. The global language selector persists the user's choice for authenticated product pages with the `courtflux_locale` cookie.

Do not translate or replace the approved CourtFlux logo and product imagery. Meaningful text should remain HTML/UI copy rather than being baked into images.

## Editing rules

1. Keep JSON valid. `web/tsconfig.json` has `resolveJsonModule` enabled, so the Next.js build imports these files directly.
2. Keep URLs as application paths such as `/login?mode=signup` or `/trust` unless an external destination is intentionally required. Public links are localized automatically when a localized route is rendered.
3. When changing English public copy, add the matching exact-string translations to both locale dictionaries or the relevant scoped translation overlay in the same pull request.
4. Use European basketball terminology: Portugal Portuguese, not Brazilian Portuguese; Spain Spanish, not Latin-American Spanish.
5. Do not claim roadmap capabilities as live. P2 basketball-state/event outputs may be described as current structural capability, but formal correctness claims must wait for reviewed human ground truth.
6. Keep the 60-day direct-upload source-video retention rule consistent across public pages. Derived analysis results/statistics remain after source deletion.
7. Do not publish `academies.json` as a route until the Academy architecture/product scope in roadmap #16 is ready.
8. Brand assets, component logic, accessibility markup, responsive layout and authenticated/product state stay in React/CSS. JSON is for editable public-page content and supported composition variants, not application logic.

After editing JSON or translations, run the normal Web quality workflow (typecheck, lint and build) before merging.
