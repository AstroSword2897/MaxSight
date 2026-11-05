# MaxSight V1 hazard taxonomy

- **Status: PROPOSED (v0.9.0). Not approved.** Ticket 301 / DR-202. It becomes v1.0 only after owner sign-off.
- **Machine-readable source:** [configs/hazard/taxonomy.yaml](../../configs/hazard/taxonomy.yaml), checked by
  `tests/unit/test_hazard_taxonomy.py`. If this page and the YAML disagree, the YAML wins and this page is wrong.
- **Builds on:** DR-018 (four categories, Confirmed), DR-019 (urgency 0–3, Confirmed), DR-020 (perception states),
  DR-023 (urgency by interpretable rules), D3.4 (category assignment and rules R1–R7).
- **Nothing reads this file at run time yet.** `configs/data/label_map.yaml` (the 17 detector labels) is unchanged.

## Three separate fields

| Field | Answers | Set by | Values |
|---|---|---|---|
| **class** | What is it? | Annotation (human), or the detector via its label | 7 hazard classes (below) |
| **category** | How is it behaving now? | Hazard assessor, per frame (D3.4) | dynamic, static, environmental, context_dependent |
| **urgency** | How soon must the user know? | Rule table R1–R7, per frame (DR-023) | 0 informational, 1 caution, 2 high, 3 critical |

Rules that keep them apart:

1. A class never fixes an urgency. A parked bicycle in the corridor and a bicycle three metres to the side share a
   class but not an urgency.
2. A class lists the categories it *may* take. The category actually used comes from motion and path evidence. For
   example, a person is `dynamic` when approaching, `static` when standing in the path, and `context_dependent`
   outside it.
3. Unknown is never a number. Missing or insufficient evidence is a perception state (DR-020:
   INSUFFICIENT_EVIDENCE, PERCEPTION_UNAVAILABLE). It is never urgency 0, and annotations never carry urgency at
   all (ticket 304 enforces this in the gold schema).

## Classes

**Critical** means a miss can cause a fall or a high-energy collision. Critical classes get the stricter coverage
gate (DR-113 proposal, G2) and 100 % second-person review in the safety-evaluation set (ticket 307).

| Class | Critical | May be | Include | Exclude |
|---|---|---|---|---|
| `curb_dropoff` | **Yes** | environmental, context_dependent | Kerb edges; platform and dock edges; a step off a raised path or ramp side | Painted kerbs with no height change; flush kerb cuts; full flights (→ stairs) |
| `stairs` | **Yes** | environmental, context_dependent | Indoor and outdoor flights, up or down; two or more steps; stopped escalators | A single kerb step (→ curb_dropoff); ladders; pictures of stairs |
| `static_obstacle` | **Yes** | static, context_dependent | Furniture, bins, poles, bollards, hydrants, cones, boxes or bags in a walking path; low or head-height obstacles | Walls and facades; objects behind glass or barriers; flat ground markings |
| `door` | No | context_dependent | Hinged, sliding and revolving doors, open or closed; gates in a path | Windows; cupboard and vehicle doors |
| `person` | No | dynamic, static, context_dependent | Pedestrians, standing or moving; people seated in the path | People in posters, screens or reflections; riders (an ignore region; the vehicle is the hazard) |
| `bicycle` | **Yes** | dynamic, static, context_dependent | Bicycles moving or parked, with or without a rider; e-scooters (to confirm) | Empty bicycle racks; pictures of bicycles |
| `vehicle` | **Yes** | dynamic, context_dependent | Cars, buses, trucks, motorcycles, moving or parked across a path | Vehicles behind barriers; toy and model vehicles |

## Reconciling the 7 hazard classes with the 17 detector labels

Ticket 301 names 7 hazard classes; `label_map.yaml` has 17 detector labels (the DR-202 default). The proposal keeps
both levels: the detector predicts its 17 labels, and each label maps to exactly one hazard class. This keeps the
finer labels that public data provides, and keeps hazard logic on 7 classes.

| Hazard class | Detector labels | COCO | Open Images | Mapillary Vistas | BDD100K (eval only) |
|---|---|---|---|---|---|
| `curb_dropoff` | **none yet** | — | — | candidates to confirm on download | — |
| `stairs` | stairs | — | Stairs | — | — |
| `static_obstacle` | chair | chair | Chair | — | — |
| | couch | couch | Couch | — | — |
| | table | dining table | Table | — | — |
| | bench | bench | Bench | — | — |
| | potted_plant | potted plant | Houseplant | — | — |
| | bag_box | suitcase, backpack, handbag | Box, Suitcase, Backpack, Handbag | — | — |
| | trash_bin | — | Waste container | — | — |
| | pole_bollard | — | Street light | object--support--pole, object--support--utility-pole, object--bollard | — |
| | fire_hydrant | fire hydrant | Fire hydrant | — | — |
| | traffic_cone | — | Traffic cone (**not a boxable class**) | — | — |
| `door` | door | — | Door | — | — |
| `person` | person | person | Person | — | person (rider → ignore) |
| `bicycle` | bicycle | bicycle | Bicycle | — | bicycle |
| `vehicle` | vehicle | car, bus, truck | Car, Bus, Truck | — | car, bus, truck |
| | motorcycle | motorcycle | Motorcycle | — | motorcycle |
| *unmapped* | animal | dog, cat | Dog, Cat | — | — |

Source ignore lists (become ignore regions, never background): COCO bed, toilet, tv, refrigerator, oven, sink; Open
Images Cabinetry, Shelf, Desk, Bed, Wall; BDD100K rider.

**Coverage consequence.** `curb_dropoff` has no public source label at all, and `traffic_cone` has no usable one.
Both depend on MaxSight's own capture (ticket 306) and, for curbs, on confirming Vistas labels. Current counts are in
[DATASET_STATUS.md](../data-governance/DATASET_STATUS.md).

## Data Role Rule

This rule applies to the taxonomy and to every dataset document and pipeline.

| Data role | Purpose | Allowed use |
|---|---|---|
| Training data | Learn detector parameters | Training only |
| Validation data | Model selection and hyperparameter tuning | Selection and tuning only |
| Independent Safety Evaluation data | Measure real performance | SG-01 and SG-02 only. **Never** used for training or model selection |

## Decisions needed for v1.0

1. **Critical flags.** Proposed: curb_dropoff, stairs, static_obstacle, bicycle, vehicle are critical; door and person
   are not. Is `person` non-critical acceptable for V1?
2. **`animal`.** It maps to no V1 class. Add a class, treat it as a dynamic hazard under another class, or drop the
   detector label?
3. **e-scooters** under `bicycle`?
4. **Two-level model.** Approve the 17 detector labels mapping onto 7 hazard classes, rather than training the
   detector directly on 7 classes.
5. **Approval.** On sign-off: set `status: approved` and `schema_version: "1.0.0"` together (the test enforces
   this), then create `configs/hazard/` consumers and the per-class severity referenced in D3.4.
