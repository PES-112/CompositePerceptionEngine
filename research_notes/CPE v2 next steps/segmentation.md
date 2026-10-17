# Scene context: walkable-surface segmentation for CPE (SANPO-trained, iPhone 17 Pro Max)

Scope: the missing component in architecture.md §6.3. It produces per-track surface / in-corridor / near-boundary, the per-scene distance to the nearest walkable edge, and a walkable-ground mask for the depth sweep, running async at 2–5 Hz.
Method: 17 tool calls (web search/fetch, plus direct inspection of the public SANPO bucket via HTTPS metadata and one 24 KB mask PNG plus one annotation JSON). No large data was downloaded and no repo files were modified.

## Q1. Published SANPO segmentation results, label volume, and mask storage

### Takeaway
Published SANPO baselines are low. Real-trained mIoU on all 31 classes is 43–48 (Kmax-DeepLab), and synthetic-only training transfers badly (14.6–17.6 mIoU on real). Synthetic data helps only as pretraining or as a ≤50% mix, adding about +0.7 to +2 mIoU. The closest precedent to CPE is Google's own on-device 3-class model (quantized MobileNetV3 at 512×512): safe-for-walking IoU 83.4, not-safe 64.5. Masks are full-resolution 8-bit RGB PNGs. R holds the semantic class ID and G/B appear to hold the instance ID. Every frame has a mask, but only every 6th frame is human-annotated (2.5 Hz); the rest are machine-propagated.

### Cited Findings
**Label volume (paper):**
- SANPO-Real has 701 sessions and 617,408 stereo frames, about 30 s each at 15 FPS — [SANPO arXiv 2309.12172v2](https://arxiv.org/html/2309.12172)
- Panoptic labels cover **237 sessions**, left lens of one camera only: 146 ZED-2i (described as "chest, long-range") and 91 ZED-M ("head, wide-angle") — [SANPO paper](https://arxiv.org/html/2309.12172)
- Labelled frames: **18,787 human-annotated** and **93,981 machine-propagated** (~112K total). Object masks: 975,207, of which 195,187 are manual. Annotators labelled "every sixth frame (0-6-12-…)", i.e. 90 frames per sub-video, and AOT propagated the frames in between. Propagation quality is J&F 0.892, described as a lower bound. Instance IDs are temporally consistent — [SANPO paper](https://arxiv.org/html/2309.12172)
- SANPO-Synthetic: 113,794 frames, 1,961 sessions (960 simulated ZED-2i, 1,001 ZED-M). Frame rates are 5 FPS (60%), 14.28 FPS (20%) and 33.33 FPS (20%). Panoptic labels and depth are pixel-accurate. The data was generated with Parallel Domain — [SANPO paper](https://arxiv.org/html/2309.12172); [SANPO GitHub](https://github.com/google-research-datasets/sanpo_dataset)
- Taxonomy: 31 classes (15 thing, 16 stuff) — [SANPO paper](https://arxiv.org/html/2309.12172). Verified `labelmap.json` IDs: 0 unlabeled, 1 road, 2 curb, 3 sidewalk, 4 guard rail/road barrier, 5 crosswalk, 6 paved trail, 7 building, 8 wall/fence, 9 hand rail, 10 opening-door, 11 opening-gate, 12 pedestrian, 13 rider, 14 animal, 15 stairs, 16 water body, 17 other walkable surface, 18 inaccessible surface, 19 railway track, 20 obstacle, 21 vehicle, 22 traffic sign, 23 traffic light, 24 pole, 25 bus stop, 26 bike rack, 27 sky, 28 tree, 29 vegetation, 30 terrain — [labelmap.json](https://storage.googleapis.com/gresearch/sanpo_dataset/v0/labelmap.json)
- Train and test split lists hold mutually exclusive session IDs, in a `splits` folder for both real and synthetic. The full bucket is about 6 TB, and `.*/segmentation_masks` can be excluded from download. Licence: CC BY 4.0 (data), Apache-2.0 (code) — [SANPO GitHub](https://github.com/google-research-datasets/sanpo_dataset)

**Baselines** (all evaluated on the human-annotated frames of the SANPO-Real test split):
- Zero-shot Kmax-DeepLab trained on Cityscapes scores 38.6 mIoU on SANPO-Real (ResNet-50) and 42.4 (ConvNeXt-L), against 79.7 and 83.5 on Cityscapes. That is a 41-point egocentric domain gap — [SANPO paper, Table 1](https://arxiv.org/html/2309.12172)
- Zero-shot Mask2Former Swin-L per-class IoU (Real / Synthetic), Table 7:

  | Class | Real | Synthetic |
  |---|---|---|
  | road | 0.255 | 0.407 |
  | sidewalk | **0.120** | 0.262 |
  | terrain | 0.394 | 0.166 |
  | pedestrian | 0.679 | 0.878 |
  | vehicle | 0.658 | 0.817 |
  | pole | 0.310 | 0.586 |

  Overall mIoU is 0.417 on real, 0.476 on synthetic and 0.833 on Cityscapes. The surface classes CPE needs are exactly the ones Cityscapes-trained models handle worst on SANPO — [SANPO paper, App. A.4.1](https://arxiv.org/html/2309.12172)
- Kmax-DeepLab trained on SANPO (mIoU / PQ):

  | Training set | ResNet-50 | ConvNeXt-L |
  |---|---|---|
  | Synthetic-Train only | 14.6 / 9.6 | 17.6 / 11.1 |
  | Real-Train | 43.4 / 34.6 | 46.0 / 38.4 |
  | Real human-annotated GT only | 42.7 / 34.6 | 46.5 / 38.2 |

  — [SANPO paper, Tables 2–3](https://arxiv.org/html/2309.12172)
- Synthetic-to-real (mIoU / PQ):

  | Setup | ResNet-50 | ConvNeXt-L |
  |---|---|---|
  | Synthetic pretrain → real fine-tune | 44.1 / 34.6 | 47.0 / 38.3 |
  | Mixed, 50% synthetic | 43.0 / 34.6 | **48.1** / 38.3 |
  | Mixed, 80% synthetic | 40.7 / 31.2 | 45.2 / 35.4 |

  The caption says synthetic data "consistently improves performance", but the 80% mixes fall below real-only training. Training used ImageNet-pretrained encoders, 1089×1921 input, 60K steps and batch size 32 — [SANPO paper, Table 4](https://arxiv.org/html/2309.12172)
- On-device baseline: a quantized MobileNetV3 at 512×512 with a 3-class scheme scores IoU **safe-for-walking 83.4, not-safe 64.5, obstacles 85.9** — [SANPO paper, Table 5](https://arxiv.org/html/2309.12172)
- SAM with a centre-point prompt reaches 48.9 instance mIoU on SANPO-Real — [SANPO paper, Table 1](https://arxiv.org/html/2309.12172)

**Mask storage** (verified directly from the bucket on 2026-10-14, one session):
- Layout: `sanpo-real/<session>/camera_chest/left/segmentation_masks/000000.png …`, plus `camera_chest/left/frame_segmentation_annotation_type.json`. There is one PNG per video frame (frames are numbered contiguously) — [GCS JSON API listing](https://storage.googleapis.com/storage/v1/b/gresearch/o?prefix=sanpo_dataset/v0/sanpo-real/-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG/)
- PNG format: **2208×1242, 8-bit, colour type 2 (RGB)**, about 24 KB per frame. On frame 000006:
  - R values were {0,1,2,3,7,8,20,21,28,…}, matching labelmap IDs (building, sidewalk, unlabeled, tree, road, vehicle, wall, curb, obstacle).
  - (G,B) pairs such as (21,0,1), (21,0,2), (20,0,1) and (28,0,1) are distinct instances of thing classes, so G/B likely hold the instance ID (high byte, low byte). This decoding is my inference from one frame and is not documented — [mask 000006.png](https://storage.googleapis.com/gresearch/sanpo_dataset/v0/sanpo-real/-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG/camera_chest/left/segmentation_masks/000006.png)
- The annotation-type JSON maps each frame index to `HUMAN_ANNOTATED` or `MACHINE_ANNOTATED`. In this session there were 528 frames: **88 human and 440 machine**, with human labels at frames 0, 6, 12, … (2.5 Hz at 15 FPS) — [frame_segmentation_annotation_type.json](https://storage.googleapis.com/gresearch/sanpo_dataset/v0/sanpo-real/-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG/camera_chest/left/frame_segmentation_annotation_type.json)

### Inferences
- **Volume in the team's 53 curated sessions.** At ~500 frames per session they hold roughly 53 × ~500 ≈ 25k mask frames, of which about 4.4k are human-annotated (43 train sessions ≈ 3.6k human frames, 10 test ≈ 0.85k). Treat these as estimates; count them exactly from the JSONs.
- **More labelled data exists.** The paper reports 237 labelled sessions but the curated set has 53. The ~180 other labelled sessions (including ZED-M head-camera ones) could be used for segmentation training, keeping official-test sessions out, even though they are not in the replay set. This is the single biggest data lever.
- **Training on propagated frames.** Table 3 shows that training on human-only frames costs ≤0.7 mIoU versus human plus propagated. Training on all frames is fine, but **evaluate only on HUMAN_ANNOTATED frames**, as the paper does.
- **Camera mismatch.** The mask path says `camera_chest`, and the paper calls the ZED-2i "chest", but `docs/sanpo_dataset.md` and the task brief describe a head-mounted ZED 2i. This needs reconciling, because viewpoint height affects the horizon line and the geometry of the walkable corridor.
- **Decoding masks.** Use `class = R`. Map 0 (unlabeled) to the ignore index (255). Decode with PIL/numpy (PIL is not installed in the system Python here).
- **Expected accuracy.** The 3-class MobileNetV3 result (83.4 / 64.5 / 85.9) is the realistic ceiling to compare against for a coarse walkable scheme. Fine-grained 31-class mIoU in the 40s is expected and is not the right target.

### Gaps
- No published per-class IoU after fine-tuning for curb, crosswalk, stairs or paved trail; Table 7 is zero-shot only. This must be measured.
- The G/B instance encoding is not documented in the README or the paper. It was inferred from one frame.
- I did not check whether synthetic masks use the same RGB encoding, or exactly how many of the 462 curated sessions fall in the 237 labelled ones (the team counts 53).

## Q2. Small mobile segmentation models: accuracy and measured iPhone / Core ML latency

### Takeaway
The best-supported choice for an iPhone 17 Pro is **Ultralytics YOLO26n-sem / s-sem**, a dedicated semantic-segmentation task:
- 78.3 / 80.8 Cityscapes mIoU at 1.6M / 6.5M parameters.
- Measured on iPhone 17 Pro: **4.5 ms inference, 4.9 ms end to end at 640×640, Core ML FP16, argmax inside the graph**.
- Built-in Core ML export and a PNG-mask training format that matches SANPO's after an R-channel extraction.

SegFormer-B0 (71.9 mIoU on Cityscapes at 512 short side; 37.4 ADE20K) is a reasonable second candidate, but it has no published iPhone latency and has known Core ML pitfalls. Any of these models is far below the 2–5 Hz budget: even 30 ms per inference uses less than 15% of a 200 ms slot.

### Cited Findings
**YOLO26-sem** (`task=semantic`, `-sem` weights):

| Model | Cityscapes mIoU (val, 1024×2048) | Params | GFLOPs |
|---|---|---|---|
| YOLO26n-sem | 78.3 | 1.6M | 23.8 |
| YOLO26s-sem | 80.8 | 6.5M | 91.0 |
| YOLO26m-sem | 82.0 | 14.3M | 305.5 |
| YOLO26x-sem | 83.6 | 40.1M | — |

ADE20K at 640: n 38.8, s 45.6 mIoU — [Ultralytics semantic docs](https://docs.ultralytics.com/tasks/semantic)
- Training format: single-channel PNG masks with pixel value = class ID and **255 = ignore**. Exports to CoreML `.mlpackage` (with `quantize`) and to Apple Core AI — [Ultralytics semantic docs](https://docs.ultralytics.com/tasks/semantic)
- The docs warn that thin structures (lanes, poles) look jagged at low `imgsz`, and advise testing at higher resolution before export. Curbs are a thin structure — [Ultralytics semantic docs](https://docs.ultralytics.com/tasks/semantic)
- **Measured on device:** YOLO26n-sem on **iPhone 17 Pro, iOS 27.0**, 640×640, 19 classes:

  | Runtime | Inference | Total |
  |---|---|---|
  | Core ML FP16, argmax in graph | 4.52 ms | **4.86 ms** |
  | Core ML INT8 | 4.60 ms | 4.95 ms |
  | Core AI FP16, logits + Swift argmax | 3.47 ms | 8.55 ms |

  For Core AI, the Swift argmax over 19 Float32 planes costs about 3.8 ms. An in-graph argmax under Core AI hardware acceleration gave **wrong output** (198,442 of 409,600 pixels differed from CPU), reported upstream as apple/coreai-torch#116 — [ultralytics/yolo-ios-app #322](https://github.com/ultralytics/yolo-ios-app/issues/322)
- A third-party paper (GTR) compares against YOLO26-sem on Cityscapes: GTR-S 81.5 vs YOLO26s 80.8 — [GTR arXiv 2609.26590](https://arxiv.org/pdf/2609.26590) (seen via search snippet only)

**SegFormer:**
- SegFormer-B0 has 3.8M parameters and 8.4 GFLOPs. It scores 37.4 mIoU on ADE20K, and **71.9 mIoU at 47.6 FPS** on Cityscapes with 512 short side (desktop GPU) — [SegFormer arXiv 2105.15203](https://arxiv.org/pdf/2105.15203) (via search summary)

**EfficientViT-Seg** (MIT Han Lab, Cityscapes at 1024×2048):

| Model | mIoU | Params | MACs | Jetson Nano | Jetson Orin |
|---|---|---|---|---|---|
| B0 | 75.65 | 0.7M | 4.4G | 275 ms | 9.9 ms |
| B1 | 80.55 | 4.8M | 25G | — | 24.3 ms |

B1 on ADE20K: 42.8 mIoU. No iPhone numbers published — [EfficientViT-Seg README](https://github.com/mit-han-lab/efficientvit/blob/master/applications/efficientvit_seg/README.md); [EfficientViT ICCV'23](https://arxiv.org/pdf/2205.14756)

**Real-time CNNs** (FPS figures come from different GPUs, so they are not comparable):
- PIDNet-S: 78.8 Cityscapes mIoU, 7.6M params — [PIDNet arXiv 2206.02066](https://arxiv.org/pdf/2206.02066) (number via a later paper's table in search results)
- PP-LiteSeg-T1: 72.0 mIoU at 273.6 FPS; PP-LiteSeg-B2: 77.5 mIoU at 102.6 FPS (GTX 1080Ti) — [PP-LiteSeg](https://www.alphaxiv.org/abs/2204.02681v1)
- BiSeNetV2-L: 75.8 mIoU — [GCNet arXiv 2503.03325](https://arxiv.org/pdf/2503.03325) (via search)
- TopFormer-T: 36.1 ADE20K mIoU at 1.2 GFLOPs (search snippet) — [GCNet arXiv 2503.03325](https://arxiv.org/pdf/2503.03325)

**Other iPhone Core ML segmentation data points:**
- Apple's DETR-ResNet50 semantic segmentation, FP16: 40 ms on iPhone 15 Pro Max, ANE-dominant — [apple/coreml-detr-semantic-segmentation](https://huggingface.co/apple/coreml-detr-semantic-segmentation)
- DeepLabV3 (Core ML): 29 ms on iPhone 12 Pro, 240 ms CPU-only — [tucan9389/SemanticSegmentation-CoreML](https://github.com/tucan9389/SemanticSegmentation-CoreML)
- iFormer reports that PVTv2-B (spatial-reduction attention, the same family as SegFormer's MiT encoder) needs 27 ms for ADE20K segmentation, against 6.6 ms for iFormer-L, on its mobile benchmark device — [iFormer arXiv 2501.15369](https://arxiv.org/pdf/2501.15369) (via search snippet)

**Phone deployment in assistive work (SENSATION-DS):**
- Android/ONNX on "Snapdragon-class" phones, 512×384:
  - SegFormer-MiT-B0: 3.8 FPS
  - DeepLabV3+-MobileNetV3: 7.4 FPS
- This was CPU/ONNX, not an NPU — [SENSATION-DS arXiv 2607.21137](https://arxiv.org/html/2607.21137v1)

### Inferences
- **Recommended model: YOLO26s-sem** (n-sem if power matters), trained at about 640–1024 on the long side.
  - Latency is measured on almost the target phone (17 Pro vs 17 Pro Max, same A19 Pro class).
  - It has a first-party Core ML export with the class map computed in the graph.
  - Its PNG-mask format takes SANPO masks after `R → remap → uint8`.
  - At 2–5 Hz, s-sem should still be about 10–20 ms on the ANE. This is an extrapolation from n-sem's 4.5 ms and the ~4× FLOPs, not a measurement.
- **SegFormer-B0 as the ablation or backup.** It is the architecture named in architecture.md and is well documented with HF/MMSeg training. Expect more conversion work (see Q5) and likely slower ANE execution than an all-conv model, because the attention reshapes are ANE-unfriendly.
- **EfficientViT-B0/B1** has the best accuracy per MAC on paper, but there is no iOS evidence. Treat it as research-only.
- **Latency is not the constraint at 2–5 Hz.** Because of that, CPE could afford a larger input (e.g. 768–1024 wide). Small far curbs and stairs edges need resolution more than they need speed.

### Gaps
- No published iPhone/ANE latency for SegFormer-B0, PIDNet, PP-LiteSeg, TopFormer or EfficientViT-Seg. These must be measured with the Xcode Core ML performance report.
- YOLO26-sem latency is only published for n at 640², on the iPhone 17 Pro rather than the Pro Max.
- MobileViT+DeepLabv3 ADE20K numbers were not retrieved.

## Q3. Sim-to-real with SANPO-Synthetic, pretraining, and label-space remapping

### Takeaway
On SANPO itself, synthetic data alone is nearly useless (14.6–17.6 mIoU on real). The safe recipe is:
1. Pretrain on a large street dataset (Cityscapes or Mapillary Vistas; YOLO26-sem ships Cityscapes weights).
2. Optionally do a synthetic stage, or mix synthetic at ≤50% of each batch.
3. Fine-tune on SANPO-Real.

Synthetic data adds about +1–2 mIoU at best and hurts at 80%. Remap the 31 classes to a small walkable-oriented scheme before training. The SANPO authors' own on-device model used a 3-class scheme, and the SENSATION work shows that coarse taxonomies plus safety metrics are the right lens.

### Cited Findings
- **SANPO synthetic results:**
  - Synthetic-only: 14.6 (R50) and 17.6 (ConvNeXt-L) mIoU on real, against 43.4 and 46.0 real-trained.
  - Synthetic pretrain → real fine-tune: +0.7 and +1.0.
  - 50/50 mix: −0.4 and +2.1.
  - 80% synthetic: −2.7 and −0.8.

  — [SANPO paper, Tables 2 & 4](https://arxiv.org/html/2309.12172)
- **SENSATION-DS** (chest-height phone views, 9 classes), staged adaptation:
  - Direct source→target transfer hurt every source; e.g. SideGuide-pretrained mIoU fell from 0.683 to 0.406.
  - Synthetic (ControlNet) images gave the best single mIoU gain with Cityscapes pretraining (0.683 → 0.704).
  - SAM2 pseudo-labels on unlabelled target images reduced road-as-sidewalk errors in every comparison (Cityscapes 0.102 → 0.086; SANPO 0.102 → 0.085).
  - Order matters: SAM2→Synthetic gave the best mIoU, Synthetic→SAM2 the lowest false-safe rate.
  - Loss was 0.3 Dice + 0.7 CE, with AdamW and OneCycle.

  — [SENSATION-DS arXiv 2607.21137](https://arxiv.org/html/2607.21137v1)
- **SENSATION-DS with SANPO as source.** SANPO was one of the source pretraining datasets, mapped to the 9-class taxonomy with unmappable classes set to ignore. The final SegFormer-B0 reached 0.699 mIoU, road IoU 0.752, sidewalk IoU 0.864 — [SENSATION-DS](https://arxiv.org/html/2607.21137v1)
- **Mapillary Vistas** covers what SANPO's surface classes need:
  - v2.0 has 124 classes over 25k images, including curb, curb cut, sidewalk, pedestrian area, bike lane and crosswalk (plain + zebra). Curb appears in 16,886 images and covers about 1.16% of image area on average — [Mapillary blog](https://blog.mapillary.com/update/2021/01/18/vistas-2-dataset.html); [Dataset Ninja](https://datasetninja.com/mapillary-vistas-dataset)
  - Caveat: Project Sidewalk/RampNet rejected Vistas "curb cut" because it includes driveways. They advise spot-checking the crosswalk and sidewalk classes before trusting them — [RampNet issue #126](https://github.com/ProjectSidewalk/RampNet/issues/126)
- **SANPO's on-device model** used 3 classes (safe-for-walking / not-safe / obstacles) — [SANPO paper, Table 5](https://arxiv.org/html/2309.12172)

### Inferences
**Proposed label scheme (8 trained classes + ignore):**

| New class | SANPO source classes |
|---|---|
| 0 sidewalk-like walkable | sidewalk(3), paved trail(6), other walkable surface(17) |
| 1 crosswalk | crosswalk(5) |
| 2 road | road(1), railway track(19) |
| 3 curb | curb(2) |
| 4 stairs | stairs(15) |
| 5 non-walkable ground | inaccessible surface(18), water body(16), terrain(30), vegetation(29) |
| 6 obstacles / things | everything vertical: building, wall, poles, vehicles, people, … |
| 7 sky | sky(27) |
| 255 ignore | unlabeled(0) |

- Whether terrain (grass) counts as walkable is a declared design choice; I would keep it non-walkable.
- Keep curb and stairs as separate classes, because they drive the edge-distance alert. Still compute a binary "walkable = {0,1}" mask for the depth sweep.
- **Training recipe:**
  1. Start from YOLO26s-sem Cityscapes weights, which already know road and sidewalk.
  2. Fine-tune on SANPO-Real train sessions (all labelled sessions, not just the 53), with ≤50% SANPO-Synthetic per batch, or a short synthetic pre-stage followed by real.
  3. Add class-balanced sampling or loss weighting for curb, stairs and crosswalk, which are thin or rare.
  4. Optionally fold in Mapillary Vistas pedestrian-view images remapped to the same scheme. Vistas has strong curb/crosswalk coverage.
  5. Use the official SANPO-Real **validation** sessions (if a split exists in-repo) for model selection, and test once on the 10 test sessions.
- **Domain gap.** It is mostly viewpoint (egocentric, low horizon) and surface appearance, which favours real SANPO data over more synthetic data. Augment with random crops along the horizon, scale jitter, colour/blur, and a horizontal flip, which is safe for this taxonomy.
- **SAM2 pseudo-labels on the ~400 unlabelled curated sessions** are a cheap semi-supervised add-on that SENSATION found reduces false-safe errors. SANPO's propagated masks already serve a similar role for labelled sessions.

### Gaps
- No published SANPO result on a coarse walkable scheme other than Table 5, and its exact class mapping was not retrieved.
- No published comparison of Cityscapes vs Mapillary pretraining on SANPO.
- The SANPO paper's "80% synthetic" setting does not specify per-batch vs per-epoch mixing.

## Q4. Prior work on sidewalk / curb / walkable-path segmentation for blind navigation, and how usefulness was measured

### Takeaway
The assistive literature has converged on coarse taxonomies (road / sidewalk / crosswalk / curb / obstacle) and on **safety-asymmetric metrics** rather than plain mIoU. The key metric is "false-safe": road predicted as sidewalk. Curb work reports distance and direction to the curb as user-facing outputs. No reviewed work measured usefulness via end-to-end replay of alerts, which is CPE's opportunity.

### Cited Findings
**SENSATION-DS (2026):**
- 2,752 chest-height images, 9 classes.
- Primary metric is the **Road-as-Sidewalk Error Rate (R→S)**, the false-safe rate. Also reports Critical False Safe Rate (CFS), Sidewalk-as-Road rate, Safe-Walkable Precision/Recall and Navigational Risk Error.
- Finding: "the highest-mIoU model was not the safest":

  | Model | mIoU | R→S | CFS |
  |---|---|---|---|
  | UPerNet | 0.715 | 0.097 | 0.058 |
  | DeepLabV3+-MobileNetV3 | 0.690 | 0.079 | 0.054 |

- The authors state the metrics are proxies and do not constitute real-world safety evidence. They also note existing datasets lack curb transitions, surface defects and small obstacles — [arXiv 2607.21137](https://arxiv.org/html/2607.21137v1)

**Other datasets:**
- SideGuide (IROS 2020): large sidewalk dataset recorded with ZED stereo cameras from a wheelchair viewpoint, with instance polygons and disparity. One source cites more than 490K images — [SideGuide (ResearchGate)](https://www.researchgate.net/publication/350082256_SideGuideA_Large-scale_Sidewalk_Dataset_for_Guiding_Impaired_People)
- SS4Blind (Kailun Yang et al.): small task-specific sets:
  - Curb segmentation: 100 images, from "Semantic perception of curbs beyond traversability…" (ICVES 2018)
  - Crosswalk segmentation: 191 images
  - Traversable-area annotations: 600 images
  - Terrain awareness: 120 RGB-D images

  — [SS4Blind GitHub](https://github.com/elnino9ykl/SS4Blind)

**Curb alerting:**
- A wearable YOLOv8-seg model trained on a custom curb dataset. It conveys curb distance and orientation via beeps, sonification and speech, in a pilot with blind and low-vision participants — [Disability & Rehab: AT 2025](https://doi.org/10.1080/17483107.2025.2463541)

**Other related work:**
- mmWalk (multi-view walking assistance) — [arXiv 2510.11520](https://arxiv.org/pdf/2510.11520)
- Synthetic data for BLV robotic mobility aids — [arXiv 2409.11164](https://arxiv.org/pdf/2409.11164)
- Also mentioned in the search results without verification: GuideDog (arXiv 2503.12844), TBRSD (tactile paving, ICCV 2023), VIDVIP

### Inferences
**Evaluation stack for CPE §6.3:**

1. **Pixel level** (human-annotated frames of the official test sessions only):
   - mIoU on the new scheme
   - per-class IoU for sidewalk, crosswalk, road, curb and stairs
   - a boundary metric (boundary IoU or boundary F at a few-pixel tolerance) for curb and stairs
   - **false-safe rate**: road, stairs or non-walkable predicted as walkable
2. **Task level**, from GT masks + depth vs predicted masks + depth:
   - per-track surface accuracy (majority label in a strip under the box bottom)
   - in-corridor agreement
   - near-boundary agreement
   - absolute error of "distance to nearest walkable edge" in metres, plus miss rate for edges within e.g. 3 m
   - reduction in "ground-ahead" pseudo-obstacles in the depth sweep, without suppressing real obstacles (count recall of true obstacles before/after masking)
3. **System level:** rerun `tools/replay_threat_engine.py` with the context multiplier on vs off. Report changes in alert precision and recall and in time-to-alert on the official split.

This level is the novel "usefulness" measurement that prior work lacks.

### Gaps
- I did not retrieve SideGuide's exact class list and mIoU, or PathNet details (no reliable source found in this pass).
- I found no published user study tying segmentation IoU to navigation outcomes.

## Q5. Core ML conversion of SegFormer / YOLO-seg: issues, versions, ANE compatibility, quantization

### Takeaway
- **YOLO26-sem → Core ML is a first-party path** with a measured 4.9 ms on iPhone 17 Pro. INT8 gives no speedup at that size.
- **Avoid Apple Core AI's in-graph argmax** until coreai-torch#116 is fixed.
- **SegFormer converts** (HF `exporters`, coremltools) but must be FP16 to use the ANE at all. Its attention reshapes and transposes follow the patterns Apple's ANE guide says to avoid.
- **Quantization:** 8-bit weight-only or palettization is near-lossless on Apple's classification benchmarks. W8A8 helps on A17 Pro and later only when the model runs fully on the ANE, and needs quantization-aware training (QAT).

### Cited Findings
- **YOLO26n-sem quantization:** Core ML INT8 4.95 ms vs FP16 4.86 ms, so no gain. The Core ML export appends `argmax(1).to(uint8)` in the graph. Under Core AI (iOS 27), in-graph argmax is wrong with hardware acceleration, so use Core ML or argmax in Swift with vDSP — [yolo-ios-app #322](https://github.com/ultralytics/yolo-ios-app/issues/322)
- **SegFormer export via HF `exporters`:**
  - `python -m exporters.coreml --model=nvidia/mit-b2 --feature=semantic-segmentation` defaulted to Float32, which runs GPU-only and does not use the ANE: 500 ms on M1 vs 300 ms with `--quantize=float16`.
  - Even in FP16, half the time was spent in GPU kernels, apparently because the final argmax ran on the GPU.
  - Float16 is the default precision in coremltools ≥7.0.

  — [huggingface/exporters #56](https://github.com/huggingface/exporters/issues/56); [huggingface/exporters](https://github.com/huggingface/exporters)
- **Apple's ANE transformer principles:**
  - (B, C, 1, S) channels-first layout
  - `nn.Conv2d` instead of `nn.Linear`
  - split attention into per-head chunks
  - avoid reshapes and transposes, which force memory copies because the last ANE axis must be contiguous and 64-byte aligned
  - einsum `bchq,bkhc->bkhq`
  - Result: up to 10× faster and 14× less memory for DistilBERT on iPhone 13. Reference code: `apple/ml-ane-transformers`

  — [Apple ML Research](https://machinelearning.apple.com/research/neural-engine-transformers)
- **coremltools quantization:**
  - Palettization at {1,2,3,4,6,8} bits. INT4/INT8 weights and INT8 activations.
  - "Most cases" lose little accuracy at 6–8-bit palettization or 8-bit weight-only.
  - W8A8 uses the int8-int8 ANE path on A17 Pro / M4 and newer. Activation quantization is recommended only if the model runs fully or mostly on the ANE. Use per-channel scales for the ANE.

  — [coremltools overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html); [quantization perf](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html)
- **Apple's measured quantization results** (iPhone 14 Pro / 15 Pro, iOS 17; classification only, no segmentation models in the table):

  | Model | Config | Top-1 | iPhone 15 Pro |
  |---|---|---|---|
  | MobileNetV2 | FP16 | 71.86 | 0.49 ms |
  | MobileNetV2 | W8A8 QAT | 71.66 | 0.20 ms |
  | MobileViTv2 | FP16 | 78.09 | 1.36 ms |
  | MobileViTv2 | W8A8 QAT | 76.89 (−1.2) | 1.03 ms |

  — [coremltools quantization perf](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html)
- **Segmentation-specific INT8 evidence** (not Core ML): post-training INT8 of YOLOv8m-seg on a CPU–DPU platform cost 6.6% relative mAP, mostly at mask boundaries and small objects. Better calibration reduced the drop to 2.0% — [arXiv 2609.02219](https://arxiv.org/pdf/2609.02219) (via search snippet)

### Inferences
- **Ship FP16 Core ML** with the argmax and a uint8 class map in-graph (the Ultralytics default). Fix the input size, e.g. 640×384 or 1024×576 to match SANPO's 16:9 aspect, instead of square padding.
- **Skip INT8:** the speed gain is nil at this size, and the risk falls on curb and stairs boundaries, the classes that matter most. If INT8 is needed for power, use QAT and check curb boundary-F, not just mIoU.
- **For SegFormer:** convert with coremltools ≥7 (mlprogram, FP16, `compute_units=ALL`) and check in the Xcode Performance report that MiT attention lands on the ANE. If argmax falls back to the GPU, return logits and do argmax with vDSP on a downsampled map.
- **Latency is fine at 2–5 Hz.** Even 20–30 ms leaves headroom beside the <50 ms p95 perception path. Run it on a lower-priority queue so it never pre-empts detection.
- **Feed measured latency into the replay.** The latency-aware replay should delay the scene-context outputs by the measured segmentation latency plus up to one async period (200–500 ms of staleness). The context multipliers will then reflect stale masks realistically.

### Gaps
- No published iPhone 17 Pro Max numbers for any of these models. YOLO26n-sem on the iPhone 17 Pro is the closest.
- No public mIoU-before/after-quantization numbers for SegFormer or YOLO-sem under coremltools.
- The current state of HF `exporters` maintenance is unclear (the README is a few years old).
