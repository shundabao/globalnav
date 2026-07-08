# GlobNav-Bench Annotation Data

This folder contains examples and generation artifacts for the proposed
GlobNav-Bench dataset.
The goal is to evaluate global, multi-modal navigation beyond pedestrian VLN:
intent parsing, clarification, environment-driven route feasibility, and
VELMA-style instruction following over walking, driving, flights, ferries, and
ground transfers.

## Files

- `examples.jsonl`: small hand-written seed examples that define the intended
  annotation style.
- `pilot_500.jsonl`: deterministic 500-example pilot generated from global
  route skeletons.
- `schema.json`: JSON schema for benchmark records.
- `pilot_500_validation_report.json`: schema and coverage summary for the
  generated pilot.

Regenerate the pilot with:

```bash
python scripts/generate_globnav_bench.py --count 500 --seed 7
```

## Collection Strategy

1. Sample anonymized origin/destination pairs from real map queries.
   Keep two names for each place:
   `anonymized_label` for paper release and `canonical_query` for internal
   geocoding/routing.
2. Use GLOBALNAV to generate candidate segment options.
   The environment records OSRM road legs, OpenFlights connectivity, Overpass
   stop evidence, and estimated public-transit/ferry fallbacks.
3. Human annotators verify the generated route.
   They reject impossible transfers, correct terminal or station choices, and
   mark evidence as `verified`, `estimated`, or `needs_api`.
4. For local walking/driving legs, annotators write co-driver style procedural
   notes from map steps and visible landmarks.
   Example: "At the third traffic light, turn left; keep the station on your
   right."
5. For flights and ferries, annotate phase transitions rather than low-level
   vehicle control.
   The simulator can jump from departure terminal to arrival terminal after
   `board -> depart/takeoff -> cruise -> dock/land`.
6. A second annotator validates:
   endpoint correctness, clarification label, segment order, oracle action
   sequence, and whether an LLM decision agent should reasonably succeed.

## Recommended Splits

- `intent_clarification`: instructions for Experiment 2.
- `planning_feasibility`: complete routes for Experiment 1.
- `route_option_generation`: segment-level option quality for Experiment 3.
- `hybrid_follower`: procedural routes for Experiment 4 style evaluation.
- `stress_test`: hard negatives and ambiguous instructions.

## Core Labels

- `schema_version`: currently `0.2`.
- `gold_intent`: origin/destination labels and internal map queries.
- `clarification`: whether the assistant should ask a question and the gold
  question type.
- `route_annotation`: segment-level modes, evidence sources, expected options,
  and known pitfalls for LLM-only planning.
- `follower_annotation`: procedural notes, oracle actions, simulator phases,
  and success criteria for LLM decision-agent evaluation.

The examples in `examples.jsonl` are hand-written seed annotations. They are
not a final benchmark; they define the expected annotation shape.

The generated pilot uses curated global city/POI seeds plus OpenFlights
connectivity checks. It is machine-validated and ready for human review, but it
should not be treated as a final released benchmark until annotators verify each
route and follower oracle.

## Annotation UI

Run the GUI and open `/bench`:

```bash
python scripts/run_gui.py --port 8765
```

The benchmark view loads `pilot_500.jsonl`, shows the route skeleton and labels,
and lets annotators save review decisions to
`data/globnav_bench/annotations.review.jsonl`.
