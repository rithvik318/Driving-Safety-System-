# AI usage

## AI inside the system

| Component | Kind | Used for | Trained by us? |
| --- | --- | --- | --- |
| YOLO26n (Ultralytics) | pretrained object detector | front-camera road objects | no (COCO-pretrained, unchanged) |
| MediaPipe Face Landmarker | pretrained landmark model | face landmarks → head pose, eye openness | no |
| MobileNetV3-Small | ImageNet-pretrained CNN | 3-class hand state (`both_hands_on_steering`, `one_hand_on_steering`, `no_hands`) | **yes**: fine-tuned on the team's labelled images |
| IoU tracker | deterministic algorithm | identity, persistence, velocity, box growth | n/a |
| Risk engine | deterministic rules | contextual risk level, reason, alarm recommendation | n/a (hand-set prototype thresholds) |
| Post-event summary hook | optional LLM call, **off by default** | rephrasing the deterministic post-event summary | not used for any published report |

The real-time loop (perception → tracking → risk → alert) contains no LLM and no learned risk model. The risk engine is not a collision predictor.

## AI-assisted development

The team used AI coding assistants, including Claude, as an engineering copilot. They helped with:

- structuring the architecture;
- generating and refining code scaffolding;
- writing and debugging Python modules;
- reasoning about model integration;
- designing the event schema and data lineage;
- writing tests;
- inspecting logs and diagnosing errors;
- iterating on perception, tracking and risk logic;
- identifying edge cases and limitations;
- drafting documentation and the presentation;
- packaging this repository.

**Human responsibilities** stayed with the team:

- project direction;
- physically collecting and organising the data;
- choosing the models;
- architecture decisions and thresholds;
- validation and interpreting failures;
- deciding what could and could not be claimed;
- final integration and evaluation.
