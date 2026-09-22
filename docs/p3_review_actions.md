# P3.7 — Review Actions

**Status:** complete for P3 foundation development.

P3.7 provides personal organization over durable P3 review items without copying or mutating basketball truth. It is deliberately very low-dependency on final P2 accuracy: P2/P3 may be reprocessed while user organization remains attached to the historical review records that were actually seen.

## Supported actions

The product surface supports:

```text
Save / Unsave
Add / Edit / Delete note
Create / Rename / Delete tag
Tag / Untag review item
Create / Edit / Delete collection
Add / Remove review item from collection
```

All actions are personal to a user inside a workspace. Sharing, assignments, messaging, notifications and collaboration permissions remain outside P3.7.

## Truth boundary

P3.7 references durable `p3_review_items.id` values. It does not duplicate:

- event type truth;
- confidence;
- player attribution;
- source-video evidence;
- clip bytes;
- P2/P3 prediction provenance.

Deleting a tag or collection removes organization metadata only. It never deletes the review item, source event, clip or human-review correction.

## Personal metadata

Existing P3.7 tables keep personal ownership and timestamps:

```text
user_id
created_at
updated_at (where mutable)
```

The read model exposes relevant timestamps for Saved items, Notes, Tags and Collections so the UI can show when personal organization was created or changed.

## Durable views and reopenability

The review workspace uses addressable URL state:

```text
/analysis/<run_id>/review?derivation=<derivation_id>
/analysis/<run_id>/review?derivation=<derivation_id>&view=saved
/analysis/<run_id>/review?derivation=<derivation_id>&view=tag&tag=<tag_id>
/analysis/<run_id>/review?derivation=<derivation_id>&view=collection&collection=<collection_id>
```

After the first load, even the current derivation is written explicitly into the URL. This is intentional: a saved/bookmarked URL keeps pointing to the same P3 output after a later reprocessing creates a new current derivation.

The UI can switch between available historical `things_to_review` derivations for the same analysis run. Historical organization remains editable and auditable; CourtFlux does not automatically remap a saved/tagged/collected historical item onto a different derivation because that could silently claim equivalence between different model outputs.

If a URL references a deleted tag or collection, the workspace falls back to the unfiltered view instead of failing.

## Saved / tag / collection views

Organization views filter only the selected derivation while also exposing total historical usage counts for Tags and Collections.

For example:

```text
Collection: "Turnovers to revisit"
3 in this review version
8 total historical items
```

This lets a coach understand both the current-version contents and the durable history of their organization metadata.

## Security model

P3.7 tables use workspace-scoped, user-owned RLS policies. The server route also validates:

- authenticated user;
- active workspace membership;
- analysis run ownership by that workspace;
- requested P3 derivation belongs to the run/game;
- review-item commands target the selected derivation;
- tag/collection commands target objects owned by the requesting user.

The browser never receives the service-role key.

## P3.7 exit checklist

- [x] Save / Unsave is durable;
- [x] Notes are durable and editable;
- [x] Tags are durable and support complete CRUD;
- [x] Collections are durable and support complete CRUD;
- [x] Items can be added/removed from tags and collections;
- [x] Saved is a dedicated review view;
- [x] Tags are dedicated review views;
- [x] Collections are dedicated review views;
- [x] organization metadata exposes personal timestamps;
- [x] views are addressable/reopenable by URL;
- [x] historical P3 derivations are reopenable without re-analysis;
- [x] reprocessing does not silently remap historical organization;
- [x] organization actions never mutate model/source truth;
- [x] collaboration features remain out of scope.

P3.7 completion does not imply full P3 Game Record reopenability or final P2 accuracy. It completes the low-dependency personal organization layer that sits on top of durable P3 review records.
