# Experimental Deviations Log

This document records deviations, design alterations, and methodology shifts made during the lifecycle of the project.

| Date | Change | Reason |
| 2026-10-10 | Synthetic benchmark validation executed (v0.1-synthetic) | Success criterion failed: GatedModel utility does not cross zero (S*=None in ~99% resamples, MI=None) against 'zeros' and 'mean' null variants because GatedModel dampens pure noise more than zero/mean vectors, keeping S=1 accuracy above null accuracy. FixedModel exhibits MI ~0.75 (zeros) and ~0.59 (mean). Parameters left untouched per protocol awaiting direction. |
