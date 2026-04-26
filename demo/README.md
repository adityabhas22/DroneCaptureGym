# DroneKit Bridge Visual Demo

This folder contains the recordable visual demo for the submission video.

## Open The Demo

From the repo root:

```bash
python3.11 -m http.server 8765
```

Then open:

```text
http://127.0.0.1:8765/demo/dronekit-bridge-visual-demo.html
```

## What It Shows

- Left panel: LLM/OpenEnv action feed.
- Center panel: animated drone route over solar rows B4-B8 with the substation no-fly zone.
- Right panel: DroneKit/SITL command mapping, telemetry, evidence status, reward progress, and story beats.

## Exact Video Line

Use this narration:

> The submitted environment trains the decision layer in a fast deterministic OpenEnv simulator. This visual shows how the same high-level actions map to DroneKit/SITL commands: takeoff becomes `simple_takeoff`, fly-to-viewpoint becomes `simple_goto`, and return-home becomes `RTL`.

Do not say the Hugging Face Space directly controls DroneKit unless the full adapter is implemented and validated.
