# Sample Images for Real-Time Object Detection

This directory contains real, curated benchmark images included with the project to test and demonstrate the YOLO object detection capabilities without requiring manual uploads or external camera feeds.

---

## Included Sample Images

| Filename | Description | Objects Present | Expected Detections (Threshold = 0.25) |
| :--- | :--- | :--- | :--- |
| `person.jpg` | Formal portrait of two people (Zinedine Zidane & colleague) | Two prominent persons, necktie | `person` (~83%), `person` (~82%), `tie` (~29%) |
| `dog.jpg` | Classic computer vision benchmark image from Joseph Redmon (Darknet) | Dog in foreground, parked bicycle, car, and pickup truck in background | `dog` (~91%), `bicycle` (~89%), `car` (~53%), `truck` (~51%) |
| `bus.jpg` | City street with pedestrians and public transit vehicle | Multiple commuters walking next to a city bus and stop sign | `bus` (~87%), `person` (4 counts: ~86%, ~85%, ~82%, ~26%), `stop sign` (~26%) |
| `blank_texture.jpg` | Neutral gradient textured background containing no recognizable real-world objects | None | `0 detections` (Validates true negatives and absence of hallucinations) |

---

## Test Verification

These images correspond to the core automated test cases defined in `tests/test_detector.py`:
- **TC1 (`person.jpg`)**: Verifies `person` class identification.
- **TC2 (`dog.jpg`)**: Verifies multi-object identification of animal (`dog`) and vehicle (`car`/`truck`).
- **TC3 (`blank_texture.jpg`)**: Verifies that images without objects return an empty detection list.
- **TC4 (Confidence Filtering)**: Verifies that applying a strict confidence threshold of `0.90` filters out lower confidence detections (e.g. car and truck at ~0.52) while preserving high-confidence objects (e.g. dog at ~0.91).
