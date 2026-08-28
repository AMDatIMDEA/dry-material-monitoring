# Tube-axis YOLO level method

## Scope and measurand

The measurand is fill level and derived bulk volume in a straight tube. The
estimator processes saved still images after acquisition, preferring CUDA when
available and otherwise using CPU. A separate operator layer may
save deliberate C920 stills first, but it never calls the estimator on preview
frames. The method does not estimate density, compare methods, or claim
validation of a trained model or physical apparatus.

In `configured` mode, the tube ROI establishes two calibrated axial limits: the full limit
and the empty limit. With `axis: top_to_bottom`, the ROI top is full and its
bottom is empty. With `axis: bottom_to_top`, the array is reversed before the
same calculation, so the configured direction always runs from full toward
empty. Lateral ROI limits exclude non-tube image content. In explicit
`whole_image` mode, no calibrated ROI is required. The complete image is passed
to YOLO, but mask-quality calculations are cropped to the tight detected
semantic extent so laboratory background is not treated as missing mask support.
For dual-class evidence, the detected semantic top/bottom are used only as
uncalibrated provisional normalization limits. That choice and warning are
recorded. Single-class evidence cannot establish both vertical limits and does
not receive an accepted percentage in this mode.

## Per-image algorithm

For segmentation mode, let `M(r,c)` and `E(r,c)` be thresholded material and
empty support inside the ROI. For explicit bounding-box mode, the configured
semantic box extents create the same binary support arrays. Define:

```text
overlap       = M and E
material_only = M and not E
empty_only    = E and not M
unclassified  = not (M or E)
```

The raw material coverage, empty coverage, overlap, and unclassified fractions
are means over the ROI. They are reported independently and never rescaled to
sum to one.

Each image is classified as `material_only`, `empty_only`,
`material_and_empty`, or unsupported when neither role appears. Multiple
instances of one role do not invalidate the image: the highest-confidence
instance supplies its mask, while every confidence and the instance count are
retained.

For a dual-class image, each axial row `r` has material and empty occupancy as
the lateral means of
`material_only` and `empty_only`. A moving-average version is used only for
orientation and minimum-run quality gates. The quantitative boundary uses the
unsmoothed occupancy. For every row edge `b` from 0 through ROI height `H`, its
disagreement cost is:

```text
C(b) = sum(material_occupancy[r] for r < b)
     + sum(empty_occupancy[r] for r >= b)
```

The boundary is the median location among exactly tied minima. This explicitly
models empty above material along the canonical full-to-empty axis. It does not
use total mask area as a volume estimate.

The normalized interface coordinate and percentages are:

```text
q = b / H
P_material = 100 * (1 - q)
P_empty = 100 - P_material
```

For supplied capacity `V_total`:

```text
V_material = V_total * P_material / 100
V_empty = V_total - V_material
```

The volume equations are implemented through the shared record package so the
complements conserve exactly within its documented tolerance.

For a single-class image, the interface is the robust edge of that class along
the tube axis. Material support must connect to the calibrated tube bottom;
Empty support must connect to the calibrated tube top. Fractional lateral and
axial support checks reject obvious tiny or disconnected detections. These are
geometric checks, not an arbitrary minimum mask-pixel count, and prevent a tiny
blob from becoming an automatic 0% or 100% result.

The calculation is also repeated independently per supported image column.
Per-image interface spread is P90 minus P10 of those
normalized column boundaries. This reveals sloped, fragmented, or inconsistent
interfaces without changing the central result.

## Acceptance gates

An image is rejected, with machine-readable reasons, when any configured or
structural gate fails:

- neither semantic role is detected;
- a detected role confidence is below its threshold;
- a detected role has no usable mask geometry;
- single-class support is tiny, laterally implausible, or disconnected from its expected tube end;
- column-interface spread is excessive;
- the dominant row sequence is reversed or has multiple transitions; or
- present material/empty support lacks the minimum dominant-row run.

Extraction inconsistencies, missing masks, inference errors, and ambiguous raw
results are also retained as rejection reasons. A rejected image keeps its
candidate percent, coverage, boundary, confidence, and reason fields, while its
accepted percent is null.

In `whole_image` mode, `row_occupancy_threshold`, minimum dominant-row support,
minimum classified-column fraction, overlap limits, and unclassified-fraction
limits are not rejection gates. Coverage/unclassified diagnostics are computed
inside the detected semantic envelope, not over the laboratory background.
Small fractional source-image extent/area checks remain solely to reject
obviously tiny detections. Confidence, NMS/IoU inference behavior, mask
threshold, semantic mapping, orientation, majority-pattern filtering, and
group averaging remain active. Interface spread is calculated and retained as
a boundary-irregularity diagnostic, but it is not a rejection rule.

Overlap and unclassified fractions are never normalized away, but neither is a
standalone rejection reason. At an exact endpoint, a geometrically complete
single-class mask may produce 0% or 100%; a small isolated detection may not.

## Capture-group result

Inference is completed for the whole group before filtering. Each image's class
pattern is counted, and the unique most frequent pattern is normally the
majority. Images with a different pattern are rejected for aggregation even if
their own geometry was valid. In the specific six-frame 3-vs-3 tie,
`material_and_empty` wins when it is one of the tied patterns because it
contains both semantic roles; its three images are retained. Other ties remain
ambiguous and invalid.

When all six confident, geometrically sane frames are `material_only`, their
accepted aggregation values are explicitly 100% material. When all six are
`empty_only`, their accepted values are explicitly 0% material. The configured
confidence thresholds and tiny/invalid detection checks still apply. This
unanimous semantic endpoint rule is recorded through the class-pattern counts,
retained flags, and accepted per-image percentages.

Let the geometrically valid, majority-pattern percentages be `p_i`. The group
candidate is the arithmetic mean `sum(p_i) / n`; group spread is
`P90(p_i) - P10(p_i)`. One retained image is permitted, while `n` is always
recorded so it can be treated as weaker evidence later. The group is invalid if
no valid image remains or retained spread exceeds its threshold. Every image,
pattern, percentage, retained flag, filter reason, and geometric rejection
reason remains in the detail table and structured evidence.

## Diagnostics and provenance

`yolo_image_details` and per-image JSON retain source path/hash, mode, class
pattern, majority pattern, retained flag/filter reasons, resolved
class IDs/names, confidences, instance counts, four coverage fractions,
interface location/fraction/spread, candidate and accepted percentages, ROI,
axis, evidence paths, validity, and rejection reasons. The per-image JSON also
retains every row occupancy.

The group JSON retains pattern counts, accepted/rejected/total counts,
arithmetic-mean rule, robust spread, formulas,
profile/mode, semantic mapping, weights SHA-256, and invalid reasons. The common
record retains the effective configuration SHA-256, model SHA-256, Git
revision/dirty state when available, schema version, and explicitly supplied
scientific inputs. The artifact manifest hashes derived evidence. Original
source images are not modified.

## Assumptions and limitations

- Tube cross-section must be constant over the calibrated axial range; otherwise
  linear level fraction is not linear bulk volume fraction.
- The camera/tube geometry and ROI endpoints must remain fixed or be recalibrated.
- The visible interface must represent the bulk material surface; occlusion,
  glare, wall residue, meniscus-like shapes, and perspective can bias it.
- A strongly sloped surface is summarized by a robust boundary and flagged by
  spread; this does not reconstruct three-dimensional volume.
- Segmentation masks depend on model generalization and threshold choice.
- `bbox_vertical_extent` is an explicit fallback for detection-only weights and
  is less spatially informative than segmentation. It requires its own study.
- P90-P10 spread is a repeatability/consistency diagnostic, not GUM uncertainty.
- Synthetic-mask correctness and software invariants do not scientifically
  validate accuracy, precision, robustness, or fitness for a publication.
