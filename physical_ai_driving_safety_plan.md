# Physical AI Driving Safety System
## Granica × IIT Guwahati DataForge — “Bring the Physical World to AI”

> **Status:** Pre-build implementation specification  
> **Target:** 48-hour hackathon prototype  
> **Core idea:** Convert real-world driving observations into structured temporal data, fuse driver state with road hazards, trigger context-aware interventions, and learn from repeated events.

---

## 1. Project Overview

We are building a **real-time Physical AI driving safety system** using two camera streams.

### Driver-facing camera
Detect:
- Both hands / one hand / no hands
- Manual distraction
- Phone/other distracting activity where feasible
- Head direction
- Eye closure
- Drowsiness
- Duration of distraction

### Front-facing camera
Detect:
- Vehicles
- Pedestrians
- Dogs / stray animals
- Pedestrian crossings
- Potholes / road damage
- Speed breakers
- Other relevant road hazards where feasible

### Temporal reasoning
Track objects across frames and derive:
- Persistent object IDs
- Trajectory
- Approach rate
- Bounding-box growth
- Persistence
- Trajectory overlap with the projected vehicle path

### Risk fusion
Combine:
- Driver attention
- Drowsiness
- Hazard type/confidence
- Object motion
- Trajectory conflict
- Persistence

Output:

`SAFE → CAUTION → HIGH → CRITICAL`

### Intervention
Trigger an audible warning only when contextual risk is sufficiently high.

Examples:
- Distracted driver + rapidly approaching vehicle
- Distracted driver + pedestrian crossing
- Distracted driver + dog entering projected path
- Drowsy driver + approaching hazard

### Event intelligence
For important events:
- Save short evidence clips
- Store timestamp and GPS
- Store structured risk metadata
- Build a Parquet event dataset
- Detect repeated risk hotspots
- Generate post-event safety insights

The LLM/agent is **post-event**, not part of the millisecond-level safety loop.

---

# 2. Why This Fits the Challenge

The project follows the physical-world workflow:

**Physical world → observation → structured data → temporal interpretation → decision → intervention → historical insight**

The important output is not simply a dashboard or an object detector.

The system answers:

> **Does the current physical situation require intervention right now?**

Examples:

- A distant car is not equivalent to a rapidly closing car.
- A pedestrian beside the road is not equivalent to a pedestrian entering the vehicle's path.
- A distracted driver on an empty road is not equivalent to a distracted driver facing an approaching hazard.
- One closed-eye frame is not equivalent to sustained eye closure.

Therefore the project focuses on **state + time + context**.

---

# 3. Core Technical Architecture

```text
                    ┌─────────────────────────┐
                    │      DRIVER CAMERA      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ Driver Perception       │
                    │                         │
                    │ Hand state              │
                    │ Activity                │
                    │ Head pose               │
                    │ Eye closure             │
                    │ Drowsiness              │
                    └────────────┬────────────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │ DRIVER STATE  │
                         └───────┬───────┘
                                 │
                                 │
┌─────────────────────┐          │          ┌────────────────────────┐
│  FRONT CAMERA       │          │          │ GPS / TIMESTAMP        │
└──────────┬──────────┘          │          └────────────┬───────────┘
           │                     │                       │
           ▼                     ▼                       ▼
┌──────────────────┐      ┌─────────────────────────────────────────┐
│ Object Detection │      │           TEMPORAL STATE                │
│                  │      │                                         │
│ Vehicles         │      │ Tracking                                │
│ Pedestrians      │      │ Approach rate                           │
│ Dogs             │      │ Trajectory                              │
│ Road objects     │      │ Persistence                             │
│ Road damage      │      │ Trajectory overlap                      │
└────────┬─────────┘      └────────────────────┬────────────────────┘
         │                                     │
         └──────────────────┬──────────────────┘
                            ▼
                  ┌──────────────────────┐
                  │     RISK ENGINE      │
                  │                      │
                  │ Driver state         │
                  │ Hazard state          │
                  │ Temporal state       │
                  │ Context              │
                  └──────────┬───────────┘
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
        ┌─────────────────┐     ┌────────────────────┐
        │ REAL-TIME       │     │ EVENT RECORDER     │
        │ ALARM           │     │                    │
        │                 │     │ Evidence clip      │
        │ SAFE            │     │ Metadata           │
        │ CAUTION         │     │ GPS                │
        │ HIGH            │     │ Timestamp          │
        │ CRITICAL        │     │ Risk reason        │
        └─────────────────┘     └─────────┬──────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │ STRUCTURED DATASET  │
                               │                     │
                               │ Parquet             │
                               │ Event records       │
                               │ Features            │
                               │ Evidence references │
                               └──────────┬──────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │ POST-EVENT ANALYSIS │
                               │                     │
                               │ Hotspots            │
                               │ Repeated hazards    │
                               │ Patterns            │
                               │ Reports             │
                               │ LLM / Agent         │
                               └─────────────────────┘
```

---

# 4. Key Technical Principle

### YOLO = What/where

Object detector gives:

```text
class
confidence
bounding box
```

### Tracker = Which object over time

```text
vehicle #17
frame 1 → frame 2 → frame 3 → ...
```

### Temporal geometry = What is it doing?

```text
bbox growing rapidly → approaching
lateral centroid movement → crossing
trajectory entering projected path → potential conflict
```

### Risk engine = Does it matter?

This separation makes the system modular, explainable, and feasible within the hackathon.

---

# 5. Data Strategy

The project will explicitly distinguish four categories.

## 5.1 Real local / hackathon observations

Already captured:

- Driver calibration frames
- Both-hand / one-hand / no-hand examples
- Drowsiness clips
- Short front-camera sequences
- Vehicles
- Pedestrians
- Dogs
- Potholes
- Speed breakers
- Pedestrian crossings
- Normal road contexts
- GPS/timestamp/event information

This is the core physical-world evidence.

## 5.2 Public auxiliary datasets

### BDD100K — primary road/temporal dataset

Use a **small relevant subset**, not the complete dataset.

Useful for:
- Driving video
- Road-object detection
- Multi-object tracking
- Temporal sequences
- Diverse road conditions
- GPS/IMU-associated driving data

Official site: https://bdd-data.berkeley.edu/  
GitHub: https://github.com/bdd100k/bdd100k

### RDD2022 — road damage

Use for:
- Potholes
- Road cracks
- Road-damage perception

Source: https://github.com/sekilab/RoadDamageDetector

### KITTI Tracking — optional

Use only if additional tracking validation is needed.

Source: https://www.cvlibs.net/datasets/kitti/eval_tracking.php

### nuScenes — not initially required

It is useful for advanced 3D/radar/LiDAR work but is unnecessary for the first prototype.

---

# 6. Model Strategy

We will **not train everything from scratch**.

## Road detection

Start with:

```text
YOLO26n pretrained model
```

Use relevant pretrained classes such as:
- car
- truck
- bus
- motorcycle
- bicycle
- person
- other available road-relevant classes

Upgrade to a larger model only if the smaller model is clearly insufficient.

## Driver perception

Use the existing cleaned driver dataset to train a lightweight classifier for:

```text
BOTH_HANDS
ONE_HAND
NO_HANDS
```

If labels support it, extend to:

```text
PHONE
OTHER_MANUAL_DISTRACTION
```

The HAPIANet paper is a task/dataset reference. We do **not** need to reproduce its full architecture.

## Face/head/drowsiness

Use face landmarks for:
- head yaw
- head pitch
- eye openness
- temporal eye-closure features

Drowsiness is treated as a temporal state rather than a single-frame classifier.

---

# 7. Vehicle Approach Detection

We do not need a separate large “approaching vehicle” training dataset.

Sequential detections are enough to derive a relative approach signal.

Example:

```text
Frame 1: bbox area = 12,000
Frame 2: bbox area = 13,000
Frame 3: bbox area = 15,000
Frame 4: bbox area = 18,000
Frame 5: bbox area = 22,000
```

Possible features:

```text
relative_size_change
bbox_growth_rate
center_motion
trajectory_direction
persistence
```

Important:

> Unless the camera is calibrated for metric depth, call this a **relative approach rate / TTC-like proxy**, not exact physical distance.

---

# 8. Pedestrian and Dog Conflict

Detection alone is insufficient.

Use:

```text
object position
+
object trajectory
+
projected vehicle path
+
persistence
```

Example:

```text
Dog detected
      ↓
Dog moves laterally
      ↓
Trajectory approaches projected vehicle path
      ↓
Driver attention is low
      ↓
Risk increases
      ↓
Alarm
```

The same logic applies to pedestrians.

---

# 9. Driver Perception

## Hand state

Primary classes:

```text
BOTH_HANDS
ONE_HAND
NO_HANDS
```

Possible additional activities:

```text
PHONE
OTHER_MANUAL_DISTRACTION
```

## Head pose

Estimate:

```text
head_yaw
head_pitch
```

Use temporal persistence so one abnormal frame does not trigger a high-risk event.

## Drowsiness

Derive:

```text
eye openness
closed-eye duration
rolling eye-closure ratio
blink duration
```

Short blink ≠ drowsiness.

Sustained/repeated eye closure raises the drowsiness score.

---

# 10. Risk Engine

Inputs:

```text
driver_hand_state
driver_activity
head_yaw
head_pitch
drowsiness_score
distraction_duration

hazard_type
hazard_confidence
object_id
approach_rate
trajectory_overlap
hazard_persistence
```

Outputs:

```text
risk_score
risk_level
risk_reason
alarm_triggered
```

## SAFE

No meaningful conflict.

## CAUTION

Something deserves monitoring.

## HIGH

Multiple risk factors interact.

## CRITICAL

Strong contextual conflict requiring immediate warning.

Example:

```text
driver distracted
+
rapidly approaching vehicle
+
high trajectory overlap
```

→ `CRITICAL`

---

# 11. Context-Aware Alarm

Avoid simplistic rules like:

```text
IF pedestrian:
    alarm
```

Instead:

```text
IF pedestrian
AND trajectory_overlap is high
AND hazard persists
AND driver attention is low:
    HIGH / CRITICAL
```

Other examples:

```text
IF vehicle approaching rapidly
AND driver distracted:
    CRITICAL
```

```text
IF dog enters projected path
AND driver is looking away:
    HIGH
```

The alarm should explain the reason:

```text
WARNING:
Driver distraction + pedestrian conflict
```

or:

```text
CRITICAL:
Driver distracted + rapidly approaching vehicle
```

---

# 12. Event-Driven Recording

Do not store every frame indefinitely.

Use a rolling buffer:

```text
continuous video
      ↓
5-second rolling buffer
      ↓
risk event
      ↓
save 5 sec before + event + 5 sec after
```

Example:

```text
events/
  event_0047/
    front.mp4
    driver.mp4
    metadata.json
```

This produces compact, meaningful evidence.

---

# 13. Event Dataset

Primary file:

```text
physical_risk_events.parquet
```

Schema:

```text
event_id
timestamp
gps_lat
gps_lon

driver_hand_state
driver_activity
head_yaw
head_pitch
eye_closure_ratio
drowsiness_score
distraction_duration

hazard_type
hazard_confidence
object_id

approach_rate
trajectory_overlap
hazard_persistence

risk_score
risk_level
risk_reason
alarm_triggered

data_source
evidence_path
```

Also maintain:

```text
dataset_manifest.json
```

with:
- collection window
- row count
- source breakdown
- schema
- observed/inferred/synthetic breakdown
- evidence paths

---

# 14. Data Provenance

Every observation is explicitly labelled:

```text
real_local
public
inferred
synthetic
```

Examples:

```text
local camera observation → real_local
BDD100K frame → public
approach_rate → inferred
generated rare combination → synthetic
```

Derived features are not presented as raw observations.

---

# 15. Simulated Data

Synthetic data will **extend real observations**, not replace them.

We do not need fake driving videos.

Instead, create feature-level stress tests based on real/public trajectories.

Examples:

```text
slow approach
medium approach
rapid approach

stationary
crossing
approaching
moving away

attentive + hazard
distracted + hazard
drowsy + hazard

no hands + approaching vehicle
phone + pedestrian crossing
drowsiness + dog entering path
```

Use synthetic data for:
- rare combinations
- threshold testing
- edge cases
- risk-engine coverage

Never claim synthetic-only evidence as real-world collection.

---

# 16. Post-Event Agentic Layer

The LLM/agent runs **after events are stored**.

Workflow:

```text
event
  ↓
structured record
  ↓
multiple events
  ↓
agent analyzes history
  ↓
repeated patterns
  ↓
precautionary report
```

Possible findings:

```text
Repeated pedestrian conflicts near one route segment.

Multiple high-risk events involved road-surface
irregularities at the same location.

Manual distraction events increased during
particular time windows.
```

The agent must reason only from recorded data and distinguish observed, inferred, and synthetic information.

---

# 17. GPS Hotspots

Every event stores:

```text
latitude
longitude
timestamp
```

Repeated events can be aggregated into risk hotspots.

Possible hotspot categories:

```text
pedestrian conflict
road damage
speed breaker
vehicle approach
driver distraction
```

This turns individual events into operational insight.

---

# 18. Repository Structure

```text
physical-ai-driving-safety/
│
├── README.md
├── ARCHITECTURE.md
├── DATASET.md
├── requirements.txt
├── .env.example
│
├── app/
│   ├── config/
│   │   └── settings.py
│   ├── driver/
│   │   ├── hand_state.py
│   │   ├── activity.py
│   │   ├── face_landmarks.py
│   │   ├── head_pose.py
│   │   └── drowsiness.py
│   ├── road/
│   │   ├── detector.py
│   │   ├── pothole.py
│   │   └── road_context.py
│   ├── tracking/
│   │   ├── tracker.py
│   │   ├── trajectory.py
│   │   └── approach.py
│   ├── risk/
│   │   ├── features.py
│   │   ├── scorer.py
│   │   └── rules.py
│   ├── events/
│   │   ├── ring_buffer.py
│   │   ├── recorder.py
│   │   └── event_store.py
│   ├── sensors/
│   │   ├── gps.py
│   │   └── timestamp.py
│   ├── analytics/
│   │   ├── hotspots.py
│   │   └── reports.py
│   ├── dashboard/
│   │   └── app.py
│   └── main.py
│
├── models/
│   ├── road/
│   ├── driver/
│   └── face/
│
├── data/
│   ├── raw/
│   │   ├── local/
│   │   ├── public/
│   │   └── synthetic/
│   ├── processed/
│   └── events/
│
├── outputs/
│   ├── evidence/
│   ├── reports/
│   └── plots/
│
├── tests/
└── scripts/
    ├── prepare_data.py
    ├── train_driver_model.py
    ├── evaluate.py
    └── run_demo.py
```

---

# 19. Implementation Plan

We will build module-by-module.

## Phase 1 — Foundation

Build:
- repository
- Python environment
- configuration
- logging
- data contracts
- dependency checks
- GPU/CPU detection

Acceptance:

```text
Project starts successfully.
```

## Phase 2 — Driver Perception

Implement:
- hand-state classifier
- face landmarks
- head pose
- eye closure
- drowsiness features

Acceptance:

```text
driver video/image
→ structured driver state
```

Example:

```json
{
  "hand_state": "ONE_HAND",
  "head_yaw": 31.2,
  "head_pitch": 4.8,
  "drowsiness_score": 0.14
}
```

## Phase 3 — Road Perception

Implement:
- YOLO26n
- road-object filtering
- road-damage component

Acceptance:

```text
front frame
→ object class + confidence + bbox
```

## Phase 4 — Tracking

Implement:
- persistent IDs
- trajectory history
- bbox growth
- approach rate
- persistence
- trajectory overlap

Acceptance:

```text
same physical object
→ same ID across frames
```

## Phase 5 — Risk Engine

Start with transparent deterministic rules.

Acceptance:

```text
state + hazard + temporal features
→ risk level + reason
```

Example:

```json
{
  "risk_level": "CRITICAL",
  "risk_score": 0.91,
  "reason": "Driver distraction + rapidly approaching vehicle"
}
```

## Phase 6 — Alarm Manager

```text
SAFE      → no alarm
CAUTION   → visual indication
HIGH      → warning sound
CRITICAL  → urgent alarm
```

Add debouncing/cooldown.

## Phase 7 — Event Recorder

Implement:
- rolling buffer
- trigger
- pre-event evidence
- post-event evidence
- metadata

## Phase 8 — Structured Dataset

Generate:

```text
physical_risk_events.parquet
dataset_manifest.json
```

## Phase 9 — GPS / Hotspots

Implement:
- aggregation
- location clustering
- hazard counts
- time analysis

## Phase 10 — Agent

Implement post-event:
- repeated-risk summary
- location patterns
- hazard patterns
- driver-state patterns
- operational insights

## Phase 11 — Demo

Support:
- live camera
- recorded-video mode

Display:
- driver state
- road detections
- tracking
- risk
- alarm
- event ID
- GPS

## Phase 12 — Final Evaluation

Evaluate:
- driver classifier
- road perception
- temporal tracking
- risk decisions
- alarm correctness
- event recording
- dataset quality

---

# 20. Compute Strategy

First verify AMD ROCm/PyTorch support.

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print(torch.version.hip)
```

Use lightweight models first.

If GPU setup becomes a time sink, maintain CPU fallback.

The hackathon priority is the working end-to-end system, not infrastructure perfection.

---

# 21. Efficiency Principles

### Avoid storing everything

Use:

```text
ring buffer + event-triggered persistence
```

### Avoid LLM in the frame loop

Use:

```text
fast perception
→ risk engine
→ event store
→ post-event LLM
```

### Avoid training common detectors from scratch

Use:

```text
pretrained detector
+
small task-specific models
```

### Avoid downloading huge datasets unnecessarily

Use:

```text
small relevant public subset
+
local validation data
```

---

# 22. What We Are Not Building

- Not a fully autonomous driving system
- Not a certified automotive safety system
- Not a giant end-to-end neural network
- Not an LLM controlling the vehicle
- Not a synthetic-only project
- Not a dashboard-only application

The target is a **research/hackathon prototype demonstrating Physical AI from observation to intervention and structured data**.

---

# 23. Privacy and Safety

Use only consented/local test footage.

Minimize retained footage.

Avoid unnecessary publication of:
- faces
- license plates
- unrelated bystanders

Use event-driven recording rather than continuous long-term storage where possible.

Document what data was collected and why.

---

# 24. Final Demo Story

The demo should show one complete physical-world loop.

### Scenario

1. Driver looks away / becomes distracted.
2. Front camera detects a relevant hazard.
3. Tracker establishes movement.
4. Temporal features show increasing conflict.
5. Risk engine combines driver + hazard + motion.
6. System outputs `HIGH` or `CRITICAL`.
7. Audible alarm triggers.
8. Evidence is saved.
9. GPS/timestamp/metadata are stored.
10. Multiple events later produce a hotspot/repeated-risk insight.

The story becomes:

**Physical World → Data → AI → Decision → Action → Learning**

---

# 25. Data Story for the Submission

```text
REAL LOCAL OBSERVATIONS
        │
        ▼
RAW CAPTURE
        │
        ├───────────────┐
        ▼               ▼
PUBLIC AUXILIARY     LOCAL DATA
DATA                 VALIDATION
        │               │
        └───────┬───────┘
                ▼
         PERCEPTION MODELS
                │
                ▼
       TEMPORAL FEATURES
                │
                ├───────────────┐
                ▼               ▼
          INFERRED DATA     SYNTHETIC
                            EXTENSIONS
                │               │
                └───────┬───────┘
                        ▼
                RISK EVENT DATASET
                        │
                        ▼
                    ALARM
                        │
                        ▼
                GPS / HOTSPOTS
                        │
                        ▼
                POST-EVENT AI
```

---

# 26. 48-Hour Priority

## P0 — Must exist

```text
data foundation
driver perception
road perception
tracking
risk fusion
alarm
event recording
structured event dataset
```

## P1 — Important

```text
GPS
hotspots
evaluation
demo interface
```

## P2 — Nice-to-have

```text
pothole specialist
agentic analysis
advanced analytics
visual polish
```

The rule is:

> **Get an end-to-end working pipeline before adding optional features.**

---

# 27. Claude Build Strategy

Claude will be used as an implementation partner **section-by-section**, not with one giant coding prompt.

Planned prompts:

```text
Prompt 1 → Project foundation
Prompt 2 → Driver perception
Prompt 3 → Road perception
Prompt 4 → Tracking + temporal features
Prompt 5 → Road-damage component
Prompt 6 → Risk engine
Prompt 7 → Dual-camera integration
Prompt 8 → Event recorder
Prompt 9 → Parquet/event dataset
Prompt 10 → GPS + hotspots
Prompt 11 → Dashboard/demo
Prompt 12 → Final testing + documentation
```

Each prompt will specify:

```text
context
current repository state
exact objective
files to create/change
interfaces
constraints
acceptance tests
run commands
```

Claude should not rewrite unrelated working modules.

---

# 28. Engineering Interfaces

Every module should expose a clean interface.

Example:

```python
driver_state = driver_pipeline.process(frame)
```

```python
road_objects = road_detector.process(frame)
```

```python
tracks = tracker.update(road_objects, timestamp)
```

```python
risk = risk_engine.evaluate(
    driver_state,
    tracks,
    context
)
```

```python
event_recorder.handle(
    risk,
    frames,
    metadata
)
```

This makes the system modular and easier to debug.

---

# 29. Final One-Line Definition

> **A Physical AI driving safety system that combines driver attention, road hazards, and temporal object motion to detect contextual risk, intervene in real time, and turn real-world driving events into structured, actionable safety data.**

---

# 30. Immediate Next Step

Before writing application code:

1. Create the repository structure.
2. Lock the data schema and configuration contracts.
3. Set up the Python environment.
4. Verify AMD GPU / CPU execution.
5. Verify local datasets are readable.
6. Use only the required public dataset subsets.
7. Establish the first end-to-end smoke test.

Then proceed through the Claude prompts one module at a time.
