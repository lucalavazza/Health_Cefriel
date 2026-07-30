# Blocker report

- Train/test leakage absent: yes; 2,400/600 disjoint participant IDs.
- Discovery data one row per participant: yes; 3,000 rows after 12-month aggregation.
- BMI and categorical labels excluded: yes.
- Incoming arrows into age and height forbidden: yes, via causal-learn BackgroundKnowledge.
- Primary graph bootstrapped: yes; PC Fisher-z alpha=0.05.
- GCM predictions present for 600/600 held-out participants per outcome.
- Agreement comparison based on 600/600 participants per outcome; missing GCM predictions: 0.
- Representative figure contains both models and both conditions: yes.
- All output checksums match across clean runs: yes.
- Non-zero downstream intervention effects: yes.
- Remaining submission blockers: none identified by the canonical run.
