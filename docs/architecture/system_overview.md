# System overview

```mermaid
flowchart LR
    subgraph Learned models
      Y[YOLO26n<br/>pretrained detector]
      M[MediaPipe Face Landmarker<br/>pretrained]
      H[MobileNetV3-Small<br/>fine-tuned hand state]
    end
    subgraph Deterministic code
      T[IoU tracker]
      D[Driver temporal logic<br/>eye-closure window · durations · activity]
      R[Rule-based risk engine<br/>gates · hysteresis · reason]
      A[Alert layer]
      E[Event recorder + writers]
      P[Post-event analytics]
    end
    F[Front video] --> Y --> T --> R
    V[Driver video] --> M --> D --> R
    V --> H --> D
    R --> A
    R --> E --> P
```

- **Real-time loop:** perception → tracking → risk → alert. It contains **no LLM**.
- **Learned models** answer only per-frame questions: what objects are present, where the face landmarks are, and what hand state is visible.
- **Deterministic code** computes everything that depends on time or context: identity, persistence, image-space approach, sustained driver states, the risk level, the reason text, the alert and the event record.
- **Recorded video only:** the system runs on recorded video (offline replay at a sampled 10 fps, CPU). Driver and front videos run as **separate** sessions, because the real recordings are not synchronized.

Module details and interfaces: [ARCHITECTURE.md](../../ARCHITECTURE.md).
