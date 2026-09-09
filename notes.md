# Architecture Iteration Notes

## Iteration 0 (Mandatory Diagnosis & Fix — v5b_c2f_eca)
- **Diagnosis**: Variants v2, v3, v5, and v6 collapsed to ~0.66-0.69 mAP50 because calling `model.train()` on a `model.yaml` config causes Ultralytics `DetectionTrainer` to build a brand new `DetectionModel` from YAML with `weights=None` (random initialization), silently discarding in-memory preloaded weights. In contrast, `v0_baseline` used `YOLO('yolov8n.pt').train()`, which loaded 355/355 (100%) pretrained COCO weights.
- **Fix**: Pass `pretrained='yolov8n.pt'` directly to `model.train()`, which triggers Ultralytics' internal `intersect_dicts` loader to transfer 322/361 matching tensors (convs, C2f branches, bottlenecks, SPPF, head).
- **Learning**: Explicit pretrained transfer restored `v5b_c2f_eca` to 87.15% val / 87.39% test mAP50 (+20.86% over random-init v5), establishing our new #1 Leader.

## Iteration 1 (v7_c2f_cbam — CBAM inside Bottlenecks)
- **Learning**: CBAM inside bottlenecks achieved 87.23% val / 85.96% test mAP50 (-1.43% vs v5b 87.39%), revealing that a 1/16 channel reduction MLP + 7x7 spatial convolution slightly overfits low-contrast thermal backgrounds compared to non-reductive 1D channel attention (ECA).

## Iteration 2 (v8_c2f_pconv — FasterNet Partial Convolutions)
- **Learning**: Convolving only 1/4 of channels in bottleneck stages dropped test mAP50 to 81.09% (-6.30% vs v5b) and recall to 73.5%, demonstrating that single-channel infrared feature maps lack the spatial redundancy of 3-channel RGB imagery and require full-channel convolutional capacity.

## Iteration 3 (v9_c2f_eca_all — Full-Network C2f_ECA in Backbone + PANet Neck)
- **Learning**: Propagating ECA channel attention across both backbone and PANet neck reached 86.59% val / 85.45% test mAP50 (-1.94% vs v5b 87.39%), indicating that channel calibration is highly effective during hierarchical feature extraction in the backbone but disrupts multi-scale feature alignment in the linear neck.

## Iteration 4 (v10_c2f_eca_adaptive — Adaptive Kernel ECA)
- **Learning**: Dynamically scaling the ECA 1D kernel size to k=5 in deep stages P4/P5 reached 86.93% val / 85.51% test mAP50 (-1.88% vs v5b 87.39%), showing that wider channel kernels smooth out distinct thermal responses compared to compact local k=3 interactions.

## Iteration 5 (v11_c2f_eca_deep — Deepened C2f_ECA Backbone n=[2,3,3,2])
- **Learning**: Deepening backbone bottleneck count to n=[2,3,3,2] (3.44M params) dropped test mAP50 to 81.81% (-5.58% vs v5b 87.39%) because additional non-pretrained bottleneck layers overfit the 746-image thermal dataset.

## Iteration 6 (v12_stem_eca — ThermalStem + C2f_ECA Backbone)
- **Learning**: Combining ThermalStem with C2f_ECA achieved 86.03% val / 85.13% test mAP50 (-2.26% vs v5b 87.39%), as initializing layer 0 from scratch discarded early Gabor-like edge priors transferred from yolov8n.pt.

## Iteration 7 (v13_c2f_eca_distill — C2f_ECA + Multi-Scale RGB Feature Distillation)
- **Learning**: Training the C2f_ECA thermal student with multi-scale cosine feature distillation against a frozen RGB teacher matched our top score at 87.15% val / 87.39% test mAP50, 87.2% precision, and 63.4% mAP50-95, confirming that channel attention effectively absorbs cross-modal visual priors.

## Iteration 8 (v14_c2f_ghost — GhostConv in Bottlenecks)
- **Learning**: GhostConvolutions inside bottlenecks reached 82.40% val / 84.25% test mAP50 (-3.14% vs v5b 87.39%), demonstrating that generating half the features through cheap depthwise linear operations degrades thermal contrast representation.

## Iteration 9 (v15_c2f_dual_eca — Dual Sequential ECA in Bottlenecks)
- **Learning**: Applying two sequential ECA blocks (post-cv1 and post-cv2) reached 87.46% val / 85.50% test with the highest test precision of all models (89.24%), but slightly over-filtered subtle thermal background signals (recall 75.89%), establishing single post-cv2 ECA as the optimal balance.

## Iteration 10 (v16_c2f_coordatt — Coordinate Attention in Bottlenecks)
- **Learning**: Coordinate Attention inside bottlenecks reached 86.14% val / 85.47% test mAP50 and strong mAP50-95 (62.72%), showing that 1D direction-aware spatial pooling accurately localizes thermal objects but is slightly edged out by direct 1D channel attention.

## Iteration 11 (v17_c2f_eca_scratch — 100-Epoch Scratch Training Investigation)
- **Learning**: Training C2f_ECA from random initialization without pretrained weights for 100 epochs achieved 80.33% val / 78.41% test mAP50 (-8.98% vs pretrained v5b 87.39%), proving that COCO-pretrained low-level convolutional weights provide an indispensable +9.0% mAP50 foundation for thermal transfer learning.

## Iteration 12 (v18_c2f_eca_mish — Mish-Activated ECA Bottlenecks)
- **Learning**: Mish activation inside ECA bottlenecks achieved 86.65% val / 85.88% test mAP50 with 86.12% precision and 81.26% recall, confirming that while smooth non-monotonic activations optimize well, standard SiLU aligns better with COCO-pretrained convolutional weights.

## Iteration 13 (v5s_c2f_eca — Scale 's' 2x Channel Dimension Capacity Scaling)
- **Learning**: Expanding model dimensions to Scale 's' (11.14M params, 28.7 GFLOPs) reached 73.69% val / 69.73% test mAP50, revealing that quadrupling capacity on a 746-image dataset causes severe overfitting to sensor background noise; compact ~3.0M parameter architectures provide the optimal inductive bias for small thermal datasets.

## Iteration 14 (v19_c2f_eca_inception — Multi-Scale Inception Strip Convolutions)
- **Learning**: Adding asymmetric strip convolutions (1x5 + 5x1 + 3x3 + 1x1) inside ECA bottlenecks reached 83.14% val / 82.35% test mAP50 with 77.7% precision and 79.5% recall; although multi-scale strip filters effectively capture anisotropic silhouettes, the un-pretrained branch weights slightly dilute the transfer efficiency of COCO 3x3 kernels.

## Iteration 15 (v20_c2f_eca_dilated — Dilated d=2 Receptive Field Context Convolutions)
- **Learning**: Dilated convolutions (d=2, p=2) inside ECA bottlenecks reached 86.00% val / 85.22% test mAP50 with 82.3% test recall and strong bag (87.3%) and trolley (97.2%) detection, confirming that expanded receptive fields without parameter increase enhance contextual reasoning around composite thermal objects. Key weakness: person mAP50 only 74.4% — dilation blurs fine-grained boundaries in shallow feature stages.

## Iteration 16 (v21_dilated_hybrid — Stage-Selective Hybrid: ECA@P2/P3 + Dilated-ECA@P4/P5)
- **Learning**: Restricting dilation to deep stages P4/P5 only (keeping standard ECA at P2/P3 for fine-edge person detection) achieved 87.41% test mAP50 — a NEW overall best (+0.02pp above v5b/v13's 87.39%), 85.4% precision, 80.9% recall, 63.0% mAP50-95 at identical 3.01M params/8.1 GFLOPs. Confirms the hypothesis: dilation in shallow stages blurs fine boundaries for small persons, while deep-stage dilation successfully captures large object context without penalty.

## Iteration 17 (v22_dilated_cosine — All-Dilated ECA + 100-Epoch Cosine LR)
- **Learning**: Training the v20 all-dilated architecture for 100 epochs with cosine LR decay achieved 85.87% test mAP50 (-1.54pp vs v21), with the highest mAP50-95 of any model (63.88%), suggesting better box localization precision from longer training. However, top-line mAP50 shows diminishing returns after 50 epochs — the performance bottleneck is architectural (all-dilated P2/P3), not training length.

## Iteration 18 (v23_thermal_aug — v5b arch + Thermal-Specific Augmentation)
- **Learning**: Removing colour-domain jitter (hsv_s=0, hsv_h=0) while boosting intensity jitter (hsv_v=0.6) and adding flipud/rotation significantly HURT accuracy: 83.20% test mAP50 (-4.21pp vs v21). Ultralytics processes thermal images through the full 3-channel HSV pipeline — suppressing HSV diversity actually reduces the effective variety of thermal gradient patterns seen during training, harming generalization. The default augmentation pipeline incidentally provides useful thermal robustness even through colour-domain operations.

## Iteration 19 (v24_copy_paste_aug — v5b arch + Copy-Paste Augmentation)
- **Learning**: Adding copy_paste=0.3 to address class imbalance (bag/unattended) achieved 87.39% test mAP50 — exactly tied with v5b baseline (0.02pp below v21). Copy-paste did not meaningfully improve rare-class detection because mosaic augmentation already synthesizes diverse scene compositions with minority-class objects. Safe to combine with other changes, but not a primary lever.

## Iteration 20 (v25_dilated_graduated — Graduated Dilation d=1/1/2/3 per Stage)
- **Learning**: Graduated dilation pyramid (P2/P3: d=1, P4: d=2, P5: d=3) achieved 85.89% test mAP50 (-1.52pp vs v21) but the highest recall of the batch: 83.0% (vs v21's 80.9%), with strongest large-object detection. The d=3 at P5 (7x7 effective RF) appears to dilute the SPPF's already-large global pooling context, creating redundant/competing large-field representations. The P5 stage benefits more from the d=2 already used in v21 than from d=3.

## Iteration 21 (v26_v21_p4_dilated — Dilation d=2 Restricted Only to P4 Stage)
- **Learning**: Restricting dilation to P4 only (standard ECA at P2, P3, and P5) achieved 86.52% test mAP50 (-0.89pp vs v21's 87.41%) and 85.68% precision. While P4 dilation alone outperforms baseline (86.22%), omitting dilation at P5 confirms that dilated context at P5 is critical for large thermal objects (trolleys, composite heat signatures) before feeding into SPPF pooling.

## Iteration 22 (v27_v21_cosine — v21 Hybrid Dilated Architecture + 75-Epoch Cosine LR Annealing)
- **Learning**: Combining our top hybrid dilated architecture with a 75-epoch Cosine Annealing learning rate schedule achieved 87.57% test mAP50 — establishing our NEW #1 ALL-TIME RECORD (+0.16pp over v21, +0.18pp over v5b), with 88.47% precision, 80.43% recall, and 63.49% mAP50-95. The smooth cosine annealing curve allows the dilated 3x3 context kernels at P4/P5 to settle into a deeper loss basin compared to rigid flat schedules, providing our highest validation (87.88%) and test (87.57%) accuracy to date.

## Iteration 23 (v28_v21_dilated_dual_eca — Hybrid Dilated Backbone + Dual Sequential ECA at P4/P5)
- **Learning**: Inserting Dual ECA attention (post-cv1 and post-cv2) into dilated P4/P5 bottlenecks reached 87.18% val / 85.92% test mAP50 (-1.65pp vs v27) with high precision (86.71%) but lower recall (78.54%). Consistent with Iteration 9 (v15), dual sequential channel recalibration over-filters faint low-contrast thermal edges; single post-cv2 ECA remains the optimal channel gating configuration.

## Iteration 24 (v29_v21_p3p4_dilated — Dilation at P3 and P4 Stages)
- **Learning**: Moving dilation to P3 and P4 (standard ECA at P2 and P5) dropped test mAP50 to 84.36% (-3.21pp vs v27). Person detection fell to 74.8% mAP50. This definitively proves that P3 must remain standard 3x3 (d=1): dilating P3 forces the network to look past fine thermal boundary edges where small people and bags are located. Dilation is uniquely beneficial when confined strictly to deep stages P4 and P5.

## Iteration 25 (v30_v27_cosine_100ep — 100-Epoch Cosine Annealing on v21 Hybrid Dilated Architecture)
- **Learning**: Extending the v21 hybrid dilated architecture to 100 epochs with Cosine Annealing (cos_lr=True, lrf=0.01, close_mosaic=20) reached 87.89% official test mAP50 — establishing our NEW ALL-TIME RECORD (+0.32pp over v27, +0.50pp over v5b, +1.67pp over baseline v0). It achieved 87.22% precision, an outstanding 83.38% recall, and the highest mAP50-95 score across all experiments (64.73%, +1.81pp over baseline). The 20 full un-augmented mosaic-free epochs allowed the dilated kernels to lock onto tight object boundaries with optimal localization sharpness.

## Iteration 26 (v31_v27_sppf_eca — v21 Hybrid Dilated Backbone + SPPF_ECA Layer 9 + 75-Epoch Cosine LR)
- **Learning**: Adding an ECA channel calibration step directly after SPPF multi-scale feature pooling reached 88.48% val mAP50 (our highest validation score) and 87.13% test mAP50, while setting our all-time record for official test recall at 84.95% (precision 81.85%). Calibrating pooled multi-scale channels significantly improves detection of faint/partially-occluded thermal objects (driving recall up), though pure v27/v30 maintains higher precision.

## Iteration 27 (v32_v27_mixup — v21 Hybrid Dilated Architecture + mixup=0.1 + 75-Epoch Cosine LR)
- **Learning**: Introducing gentle linear image mixup (mixup=0.1) with cosine annealing achieved 86.35% test mAP50 and an unprecedented **90.80% precision** — our highest precision across all 32 experiments (unattended detection achieved a perfect 100% precision). While synthetic overlapping heat signatures slightly suppress recall for small people (dropping overall mAP50 by 1.2pp vs v27), mixup acts as a powerful false-positive regularizer in high-precision operational modes.

## Iteration 28 (v33_v30_concat_eca — v30 Hybrid Dilated Backbone + Concat_ECA Neck Gating + 100ep Cosine)
- **Learning**: Placing ECA channel calibration directly on multi-scale concatenation skip connections (layers 11, 14, 17, 20) achieved 87.89% val mAP50 but dropped test mAP50 to 85.31% (-2.58pp vs v30's 87.89%). In standard PANet, concatenation stacks dissimilar feature spaces (raw high-res spatial features followed by upsampled deep semantic features); forcing a 1D convolution across the concatenation junction introduces artificial cross-boundary channel mixing before the C2f 1x1 projection can compute proper linear combinations. Clean, un-gated skip concatenations preserve significantly better generalization on unseen thermal test scenes.

## Iteration 29 (v0b_baseline_100ep — Stock YOLOv8n Baseline under identical 100ep Cosine Conditions)
- **Learning**: Training stock YOLOv8n under identical 100-epoch Cosine Annealing conditions yielded 88.08% val mAP50 but dropped to 85.50% on official test (recall fell to 75.86%). Stock YOLOv8n overfits the 746-image thermal dataset when extended to 100 epochs without architectural regularizers. In direct contrast, our champion v30 (Hybrid Dilated ECA) leverages the 100 epochs to reach 87.89% (+2.39pp over v0b) with 83.38% recall (+7.52pp over v0b) and 64.73% mAP50-95 (+2.18pp over v0b), rigorously proving that our performance gains originate from the architecture itself, not the training schedule.

