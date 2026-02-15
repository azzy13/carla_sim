Prompt-Compliance Metrics

Matching is performed using IoU ≥ threshold (e.g., 0.5) with Hungarian assignment.

1. Semantic Precision (SP)

Definition:
The fraction of predicted boxes that correspond to prompt-valid objects.

Formula:

SP = (Number of predictions matched to prompt-valid GT)
/ (Total number of predictions)

Interpretation:
Measures how often predictions correspond to the correct semantic target.

Intuition:
High SP means few semantic false positives.

2. Semantic Recall (SR)

Definition:
The fraction of prompt-valid ground-truth objects that are matched by predictions across frames.

Formula:

SR = (Number of prompt-valid GT boxes matched by predictions)
/ (Total number of prompt-valid GT boxes)

Interpretation:
Measures how well the tracker covers all valid targets.

Intuition:
High SR means few missed targets.

3. Prompt Coverage Ratio (PCR)

Definition:
The fraction of frames in which a prompt-valid target is correctly tracked, out of all frames where it is visible.

Formula (single target):

PCR = (Frames where target is matched)
/ (Frames where target is visible)

Formula (multi-target):

PCR = (Total matched prompt-valid GT across frames)
/ (Total prompt-valid GT across frames)

Interpretation:
Measures tracking reliability over time.

Intuition:
High PCR means the tracker stays on the target consistently.

Tracking Stability Metrics
4. Semantic ID Switches (SID)

Definition:
The number of times a predicted track switches from a prompt-valid object to a prompt-invalid object (or vice versa).

Conceptual rule:

A switch occurs when:

validity(t) != validity(t−1)

where validity indicates whether the matched GT object satisfies the prompt.

Interpretation:
Measures semantic drift during tracking.

Intuition:
Low SID means tracks remain semantically consistent.

Error Metrics
5. Distractor Confusion Rate (DCR)

Definition:
The fraction of predictions that match distractor objects instead of prompt-valid objects.

Formula:

DCR = (Predictions matched to prompt-invalid GT)
/ (Total predictions)

Interpretation:
Measures how often the tracker locks onto the wrong object.

Intuition:
Low DCR means the tracker rarely confuses targets with distractors.