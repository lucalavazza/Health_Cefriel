# Blocker report

- Train/test leakage absent: yes; 2,400/600 disjoint participant IDs.
- Discovery data one row per participant: yes; 3,000 rows after 12-month aggregation.
- BMI and categorical labels excluded: yes.
- Incoming arrows into age and height forbidden: yes, via causal-learn BackgroundKnowledge.
- Primary graph bootstrapped: yes; PC Fisher-z alpha=0.05.
- Non-zero downstream intervention effects: yes.
- Clean-run numerical reproducibility: verified by canonical two-run comparison.
- Remaining submission blockers: none identified by the canonical run.
