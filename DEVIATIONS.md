# Experimental Deviations Log

This document records deviations, design alterations, and methodology shifts made during the lifecycle of the project.

| Date | Change | Reason |
| 2026-10-10 | Synthetic benchmark validation executed (v0.1-synthetic) | Success criterion failed: GatedModel utility does not cross zero (S*=None in ~99% resamples, MI=None) against 'zeros' and 'mean' null variants because GatedModel dampens pure noise more than zero/mean vectors, keeping S=1 accuracy above null accuracy. FixedModel exhibits MI ~0.75 (zeros) and ~0.59 (mean). Parameters left untouched per protocol awaiting direction. |
| 2026-10-10 | Primary null changed from zeros/mean to removed after v0.1 results | Zeros/mean synthetic vectors produced biased centroid distances causing GatedModel null accuracy to drop below pure-noise accuracy. 'removed' (X_A=None) serves as true unimodal ablation. Also fixed summary aggregation to use pooled bootstrap so point estimates lie strictly within CIs, adding across-seed std and bootstrap stability columns. |
