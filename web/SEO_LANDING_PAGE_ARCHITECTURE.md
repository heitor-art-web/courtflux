# CourtFlux SEO landing-page architecture

This document defines the public search-intent architecture for the CourtFlux Next.js marketing site. The goal is to build focused pages around real basketball video-analysis problems without creating overlapping pages that compete for the same query.

## Principles

- Keep one primary search intent per page.
- Lead with basketball language, not generic AI terminology.
- Keep current beta capabilities separate from in-development and vision features.
- Use the same underlying product truth across all pages; audience pages change workflow/context, not computer-vision quality.
- Link every landing page back to the homepage, pricing, and signup.
- Add cross-links only when the destination page exists; do not create dead navigation.
- Keep `/login`, `/account`, `/lab`, and `/api/*` out of search indexing.

## Priority architecture

| Priority | Route | Primary intent | Primary keyword theme | Recommended title | H1 direction | Primary CTA | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | `/basketball-video-analysis` | Understand software that turns game film into analysis | basketball video analysis, basketball video analysis software, AI basketball analysis | Basketball Video Analysis Software for Coaches | Basketball video analysis that starts with the film you already have | Start free beta | Built |
| P1 | `/basketball-analysis-for-coaches` | Find a coach-first workflow for reviewing game film | basketball analysis for coaches, basketball coaching video software | Basketball Analysis for Coaches | Spend less time finding the moment and more time coaching it | Start free beta | Planned |
| P1 | `/basketball-analysis-for-teams` | Evaluate a shared single-team analysis workflow | basketball team analysis, basketball team video analysis | Basketball Video Analysis for Teams | One team library for game film, evidence and staff review | Start free beta | Planned |
| P1 | `/basketball-analysis-for-academies` | Evaluate multi-team / development use cases | basketball academy analytics, basketball academy video analysis | Basketball Analytics for Academies | Connect game film to player development across age groups | Talk to CourtFlux | Planned |
| P2 | `/basketball-player-tracking` | Understand player tracking from ordinary game footage | basketball player tracking, AI player tracking basketball | Basketball Player Tracking from Game Video | Track players across basketball game footage | Start free beta | Planned |
| P2 | `/basketball-ball-tracking` | Understand CourtFlux ball detection and temporal tracking | basketball ball tracking, basketball ball detection AI | Basketball Ball Tracking from Game Video | Follow the ball through the moments that define possession | Start free beta | Planned |
| P3 | `/basketball-performance-analysis` | Broader performance-analysis discovery | basketball performance analysis, basketball performance analysis software | Basketball Performance Analysis from Video | Build performance evidence from the game film you already record | Start free beta | Planned |

## Page roles and cannibalization rules

### `/basketball-video-analysis`
The broad commercial landing page. It should explain the end-to-end CourtFlux proposition: existing footage -> player/team/ball tracking -> structured basketball evidence -> future spatial/event/tactical layers. It should not become a detailed player-tracking or ball-tracking technical page.

### Audience pages
`/basketball-analysis-for-coaches`, `/basketball-analysis-for-teams`, and `/basketball-analysis-for-academies` should answer who the workflow is for and how the product fits their operating model. They should not change the underlying tracking-quality claims by plan or audience.

### Capability pages
`/basketball-player-tracking` and `/basketball-ball-tracking` should go deeper on the computer-vision problem, evidence, limitations, and how those capabilities support later basketball analytics. They should not compete with the broad product page for generic "basketball video analysis" terms.

### `/basketball-performance-analysis`
Use later as a broader performance-analysis page once enough structured analytics exist to support the promise. Until then, keep it planned rather than publishing a thin page.

## Internal linking model

When all priority pages exist:

- Homepage -> Basketball Video Analysis, Coaches, Teams, Academies.
- Basketball Video Analysis -> Player Tracking, Ball Tracking, Coaches, Teams, Academies, Pricing.
- Coach page -> Basketball Video Analysis, Player Tracking, Pricing.
- Team page -> Basketball Video Analysis, Player Tracking, Ball Tracking, Pricing.
- Academy page -> Basketball Video Analysis, Team page, future player-development content, Pricing.
- Player Tracking -> Basketball Video Analysis, Teams, Academies.
- Ball Tracking -> Basketball Video Analysis, Player Tracking.

Use descriptive anchor text such as "basketball player tracking" rather than generic "learn more" wherever it reads naturally.

## Content requirements for every public landing page

1. Unique title and meta description.
2. One clear H1 aligned to the page intent.
3. Canonical URL.
4. Open Graph and Twitter metadata.
5. Index/follow robots metadata.
6. Relevant structured data (WebPage plus FAQPage where an FAQ is present).
7. Clear beta/in-development labels around unreleased features.
8. At least one approved CourtFlux product visual with explicit alt text.
9. Primary CTA to signup or contact intent.
10. Links back to the homepage and pricing.
11. Inclusion in `sitemap.ts` only after the page exists.
12. No invented performance claims or unsupported accuracy percentages.

## Content cluster to build later

Support the commercial pages with useful, evidence-led resources rather than thin SEO articles. Strong topics include basketball video-analysis workflows, how player tracking works, ball occlusion/reacquisition, evaluating camera footage for analysis, coach review workflows, academy player-development workflows, and benchmark/validation methodology.
