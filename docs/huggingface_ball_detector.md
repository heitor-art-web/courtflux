# Hugging Face ball detector challenger

CourtFlux keeps the frozen YOLO ball detector as the default. Hugging Face object-detection models are an **experimental P0.1 challenger backend** and use the exact same CourtFlux candidate contract, 40 px GT matching, association code, diagnostics, and quality gate.

The purpose is to answer a clean question: can a stronger/different detector architecture materially improve visible ball candidate recall without making false candidates or latency unacceptable?

## Install

Use the optional environment so normal YOLO development does not require the HF stack:

```bash
python3 -m pip install -r requirements-hf.txt
```

The integration targets native Transformers object detection. RF-DETR is supported directly by current Transformers; CourtFlux intentionally keeps `trust_remote_code=False`.

## Authentication

**Never commit a Hugging Face token.** CourtFlux does not read, persist, print, or write the token itself. Authentication is delegated to `huggingface_hub`.

For a private model, either log in with the HF CLI or provide the standard environment variable before launching Python:

```bash
export HF_TOKEN='hf_...'
```

`.env` and `.env.*` are already gitignored in this repository, but shell/keychain/secret-manager storage is preferred for production.

## Backend selection

Default / frozen baseline:

```bash
export BASKETBALL_BALL_DETECTOR_BACKEND=yolo
```

Hugging Face challenger:

```bash
export BASKETBALL_BALL_DETECTOR_BACKEND=huggingface
export BASKETBALL_HF_BALL_MODEL='Roboflow/rf-detr-medium'
```

The default accepted labels are exact normalized labels:

```text
sports ball, ball, basketball
```

Override them for a custom fine-tuned model:

```bash
export BASKETBALL_HF_BALL_LABELS='ball'
```

`ball-in-basket` / `ball in basket` is intentionally **not** accepted as a ball candidate by default. It is a useful future event class, but its box is not equivalent to the physical ball position required by P0.1.

For a pinned Hub revision/commit:

```bash
export BASKETBALL_HF_MODEL_REVISION='<hub-commit-sha>'
```

For device override (useful if an RF-DETR operation is unsupported or slow on MPS):

```bash
export BASKETBALL_HF_DEVICE=cpu
# or mps / cuda when supported by the local Torch build and model
```

## Important metric semantics

Do not compare Hugging Face/Roboflow mAP or dataset recall directly with the CourtFlux Quality Gate.

Promotion is decided only by the frozen CourtFlux benchmark:

- visible GT frame;
- at least one candidate center within **<= 40 px**;
- same V1–V5 annotations;
- micro and macro candidate recall;
- worst-video recall;
- false candidates/gameplay frame;
- not-visible candidate-frame rate;
- latency/FPS.

The YOLO `--imgsz 960` operating point remains frozen for the YOLO baseline and player tracker. A Hugging Face DETR-family backend uses its model image processor rather than Ultralytics `imgsz`; this is recorded in run metadata as `image_size_semantics=model_processor` and must be considered in performance comparisons.

## Experiment RF1 — detector-only challenger

First isolate the detector. Keep reacquire OFF and disable the smart crop/ranking layer:

```bash
export BASKETBALL_BALL_REACQUIRE=0
export BASKETBALL_BALL_DETECTOR_BACKEND=huggingface
export BASKETBALL_HF_BALL_MODEL='Roboflow/rf-detr-medium'
export BASKETBALL_HF_BALL_LABELS='sports ball'
export BASKETBALL_SMART_BALL_PIPELINE=0
```

Then run the same five frozen clips with the existing analyzer commands (`--conf 0.10 --imgsz 960 --device mps --detection-only --no-jersey --no-persist`). The `--imgsz` still applies to player YOLO; the HF ball model uses its processor-native image sizing.

Use `evaluation/smart_ball_benchmark.py` with each new run mapped to the original frozen GT run. Despite the helper's historical name, it evaluates the production candidate stream and can therefore benchmark a plain HF backend too.

### RF1 decision

Do not promote RF-DETR because it has better generic mAP. Promote it only if the CourtFlux metrics improve materially, especially micro recall and worst-video recall, with tolerable candidate noise and runtime.

## Experiment RF2 — HF detector + current smart pipeline

Only after RF1 tells us the detector's standalone value:

```bash
export BASKETBALL_SMART_BALL_PIPELINE=1
export BASKETBALL_BALL_VERIFY=1
export BASKETBALL_BALL_SELECTIVE_ZOOM=1
export BASKETBALL_BALL_EMIT_REFINEMENTS=0
export BASKETBALL_BALL_ZOOM_MIN_GEOMETRY=0.60
export BASKETBALL_BALL_TEMPORAL_RESCUE_MIN_SCORE=0.70
```

The verification and temporal crop passes reuse the **same active backend**. With Hugging Face selected, the second-pass crops also go through the HF model; there is no hidden YOLO fallback.

## Private CourtFlux model on the Hub

The backend is already compatible with a future private fine-tuned checkpoint. A promoted model should be referenced by model id **and pinned revision**:

```bash
export BASKETBALL_HF_BALL_MODEL='YOUR_HF_USER_OR_ORG/courtflux-ball-rfdetr-v1'
export BASKETBALL_HF_MODEL_REVISION='<commit-sha>'
export BASKETBALL_HF_BALL_LABELS='ball'
```

The run summary records the backend, model id, revision, accepted labels, matched model labels, device, threshold semantics, and image-size semantics. The token is never included.

## Recommended training/data direction

When we start fine-tuning, build the first CourtFlux dataset version from the existing frozen material rather than collecting new videos immediately:

- visible GT ball frames;
- baseline true positives;
- no-candidate false negatives;
- >160 px false-negative cases;
- oracle-crop recoveries;
- smart temporal-zoom recoveries;
- hard negative candidates from wrong objects;
- not-visible frames with false detections;
- especially V2 representation failures and V4 high-noise cases.

Keep dataset version, training config, model checkpoint/revision, Git commit, and frozen CourtFlux benchmark together for every experiment.

## Tests

The adapter and smart-backend bridge have tests that do not require downloading a Hub model:

```bash
python3 evaluation/ball_candidate_backend_test.py
python3 evaluation/smart_ball_backend_integration_test.py
python3 evaluation/ball_candidate_pipeline_test.py
python3 evaluation/smart_candidate_filter_test.py
```

A real HF smoke run requires `requirements-hf.txt`, network access for the first model download (unless cached), and authentication if the selected model is private.
