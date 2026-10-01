# SheAlert — FYP Panel Presentation & Defense Cheatsheet

> **Purpose:** A high-impact executive reference designed for FYP panel presentations, supervisor reviews, and viva defenses. It provides instant answers, exact architectural numbers, and defensible technical justifications without needing to inspect raw code files.

---

## 1. The 30-Second Elevator Pitch

> *"Most women safety apps fail when they are needed most because they assume the victim can press a button or shake their phone. In real-world violent encounters, victims experience tonic immobility (fear paralysis), are physically restrained, or have their phone snatched. SheAlert is a **fully passive**, **on-device** safety intelligence system that continuously fuses motion and audio in the background. By combining lightweight multi-tier filtering, fine-tuned Urdu distress models, a context-adaptive late fusion engine, and a two-stage cancellation window with a covert duress PIN, SheAlert delivers high-confidence automatic SOS dispatch with zero user action required during an emergency."*

---

## 2. Vital System Statistics at a Glance

| Metric / Dimension | Exact Value | Verification / Reference |
|---|---|---|
| **Raw Harvested Audio Clips** | **31,956 clips** across 10 datasets | Pre-deduplication scan |
| **Clean Audio Retained Post-QC** | **30,264 clips** ($94.7\%$) | Passed duration, SNR, & dynamic clipping |
| **QC Rejection Breakdown** | $1,692$ rejected ($1,307$ low SNR, $323$ clipped, $62$ too short) | Documented in `output/rejected_clips.csv` |
| **Training Manifest Volume** | **22,368 clips** | Subsampled ambient to satisfy Section 10 rule |
| **Training Class Balance** | **Distress: 33.41%**, **Aggression: 29.65%**, **Normal: 36.93%** | Normal capped strictly $\le 40.0\%$ |
| **Rescued High-Energy Screams** | **688 scream clips** rescued via dynamic clipping | Audited via `analysis/clipping_check.py` |
| **Cross-Validation Folds** | **5 Folds: [4458, 4458, 4458, 4458, 4459]** | Deviation: **0.02%** (Seed 11 local search) |
| **Group / Speaker Leakage** | **0.00% (Strictly Zero Cross-Fold Leakage)** | Verified by `test_zero_group_leakage` |
| **Dataset Representation** | **All 9 non-OOD datasets present in every fold** | Verified by `test_every_non_ood_dataset...` |
| **Isolated OOD Benchmark** | **RealWorld-Urdu (77 clips)** | Pure zero-shot out-of-distribution evaluation |
| **5-Fold Cross-Validation Macro F1** | **75.91% (+/- 1.93%)** (Best Fold: **78.61%**) | Trained across 22,291 clips in 5 folds |
| **Distress Recall (Safety Metric)** | **67.96% (+/- 3.80%)** | Safety-weighted loss ($w_{\text{Distress}} = 1.2$) |
| **FP16 TFLite Model Size (LM-5)** | **1,060 KB (1.03 MB)** | Target was $\le 16\text{ MB}$ (15x smaller!) |
| **On-Device Inference Latency** | **0.068 ms** per prediction | Target was $\le 200\text{ ms}$ (2,900x faster!) |
| **Automated Test Coverage** | **31/31 Pytest suites passing** | Execution time ~21s |

---

## 3. Top 10 Panel Questions & Defensible Answers

### Q1: "Why build a passive system? Isn't a panic button or shake trigger sufficient and simpler?"
* **Defense:** "Trauma psychology literature demonstrates that high-stress violent assaults frequently induce *tonic immobility*—an involuntary neurobiological paralysis where the victim physically cannot move or scream. Furthermore, in physical grabs, an assailant immediately neutralizes the victim's hands or seizes the device. A safety system that relies on active manual triggers fails in the most violent encounters. SheAlert's passive background detection protects the user even when they cannot react."

### Q2: "Continuous audio AI burns battery quickly. How can an Android phone run this 24/7?"
* **Defense:** "We do **not** run YAMNet continuously. We implemented a 3-tier cascaded architecture:
  1. *Tier 1 (WebRTC VAD + RMS):* Operates at milliwatt power budgets to filter out silence and stationary background noise.
  2. *Tier 2 (DSP Heuristics):* Analyzes pitch variability and spectral flux to discard ordinary conversational Urdu speech.
  3. *Tier 3 (YAMNet DL):* Deep neural network inference is **dormant** over $90\%$ of the time, triggering only when anomalous acoustic spikes are verified by Tiers 1 and 2.
  Additionally, geofencing (FR-3.6) idles the intensive monitoring when the user is inside their safe home perimeter."

### Q3: "Why choose Late Fusion over Early Feature Concatenation?"
* **Defense:** "Sensory modalities fail independently in real-world mobile environments:
  * A phone inside a thick handbag muffled against acoustic input still experiences severe acceleration spikes from a struggle or fall.
  * A user shouting for help while sitting in a car has clear acoustic signals but static motion sensors.
  Early fusion concatenates feature vectors into one model, where one degraded channel corrupts the entire prediction. Late fusion keeps the motion and acoustic classifiers decoupled ($Threat = W_1 \cdot s + W_2 \cdot a$), enabling dynamic context-adaptive re-weighting and robust single-modality fallback."

### Q4: "How do you prove your cross-validation results aren't memorizing specific voices?"
* **Defense:** "Standard random K-Fold splits suffer from severe *speaker leakage*, where identical speakers appear in both training and test sets, artificially inflating test accuracy. We solved this using a **Group-Stratified 5-Fold Partition**:
  * Every actor/speaker (across CREMA-D, RAVDESS, SEMOUR+, UrduSpeech) was assigned a unique global `group_id`.
  * We designed a greedy bin-allocation algorithm followed by local swap optimization (Seed 11) that achieved equal fold sizes (4,458 clips each) with **zero speaker overlap across folds** (tested and verified with 0 group leakage)."

### Q5: "Why did you rescue 'clipped' audio files? Isn't clipped audio bad data?"
* **Defense:** "Standard audio clipping filters check for sample peaks exceeding $\pm 0.99$. In emergency acoustics, genuine high-arousal human screams naturally possess intense dynamic range and crest factors that peak near ADC headroom. Flatly rejecting these files gutted our most authentic scream datasets (*SEMOUR+* and *AudioSet Screaming*).
  We formulated a relative peak boundary: $\text{Threshold} = \text{Peak} - 0.05 \cdot (\text{Peak} - \text{RMS})$ requiring at least $0.5\%$ of samples to plateau. This rescued **688 authentic distress cries** while continuing to discard genuine flat-top square-wave noise."

### Q6: "Why is the Normal class capped at 40% in your training manifest?"
* **Defense:** "In raw environmental datasets like UrbanSound8K and ESC-50, non-threat sounds represent over $53\%$ of the total data. Training deep neural networks on predominantly Normal data biases the decision boundary toward the majority class, causing dangerous false negatives (missed distress events). By subsampling ambient audio down to 800 balanced clips, we reduced the Normal fraction to $36.93\%$, satisfying our proposal's Section 10 cap ($\le 40\%$) and enforcing high sensitivity toward Distress."

### Q7: "What if an attacker grabs the phone and cancels the alert?"
* **Defense:** "We designed a **Two-Stage Confirmation Window with a Duress / Dead Man's Switch PIN**:
  * Stage 1 (0–10s) is completely silent with discreet haptic vibration, allowing the user to abort false alarms unnoticed.
  * Stage 2 (10–20s) activates an intense acoustic siren and camera strobe.
  * If an assailant threatens the user to unlock the phone and cancel the alert, entering the secret **Duress PIN** displays a spoofed 'Alarm Cancelled' confirmation screen while silently transmitting an expedited, high-priority emergency alert to contacts and police."

### Q8: "What happens if the phone camera is covered by clothing or the pocket?"
* **Defense:** "The camera verification subsystem (Module 7) is engineered with **strict fail-safe semantics**. It spends a maximum of 2–3 seconds attempting to detect threats via MobileNet SSD. If the lens is occluded, lighting is zero, or camera permissions fail, the system **never aborts or delays** the SOS. The dispatch pipeline proceeds immediately with GPS and audio telemetry."

### Q9: "Why separate Stage 1 and Stage 2 fine-tuning?"
* **Defense:** "Urdu-specific distress recordings are rare and precious. If mixed into the initial 30,000-clip external dataset, the model would overfit on the tiny Urdu slice while training on general audio.
  In Stage 1, YAMNet learns broad acoustic distress representations across 10 diverse public datasets. In Stage 2, the network is calibrated specifically on the authentic Urdu distress corpus. This two-stage transfer learning pipeline prevents catastrophic forgetting and maximizes Urdu phonetic sensitivity."

### Q10: "What if the user is in a basement with zero internet connectivity?"
* **Defense:** "First, all AI inference is $100\%$ on-device and requires zero connectivity. Second, for alert delivery (Module 8), while Firebase Cloud Messaging (FCM) via internet is primary, if backend acknowledgment is not received within 60 seconds, the Android telephony layer automatically falls back to direct SMS cellular broadcast containing cached GPS coordinates."

---

## 4. SheAlert vs. Existing Safety Systems

| Feature | Standard Safety Apps (e.g. bSafe, Life360) | Hardware Panic Buttons | SheAlert (Our System) |
|---|:---:|:---:|:---:|
| **Trigger Mechanism** | Manual button tap / screen unlock | Physical button press | **Fully Autonomous (Passive Motion + Mic)** |
| **Incapacitation Protection** | ❌ Fails under panic / restraint | ❌ Fails if button snatched | ✅ **Operates without victim touch** |
| **Acoustic Intelligence** | ❌ None | ❌ None | ✅ **Urdu Distress + Screaming Detection** |
| **Battery Optimization** | ❌ Drains fast if continuous | N/A (Dedicated battery) | ✅ **3-Tier Cascade + Geofencing** |
| **Coercion Resistance** | ❌ Can be cancelled by attacker | ❌ Attacker throws away device | ✅ **Covert Duress / Dead Man's PIN** |
| **Network Resilience** | ⚠️ Internet dependent | ⚠️ Proprietary cellular / Bluetooth | ✅ **On-Device AI + FCM with SMS Fallback** |

---

## 5. Quick Codebase Tour for Panel Review

If the panel asks to inspect the actual code:
* **Taxonomy & Paths:** [`shealert_audio_pipeline/config.py`](file:///f:/SheAlert-Women-Safety-System/shealert_audio_pipeline/config.py)
* **Signal Processing & Clipping Math:** [`shealert_audio_pipeline/utils.py`](file:///f:/SheAlert-Women-Safety-System/shealert_audio_pipeline/utils.py)
* **Dataset Consolidation & Subsampling:** [`shealert_audio_pipeline/consolidate.py`](file:///f:/SheAlert-Women-Safety-System/shealert_audio_pipeline/consolidate.py)
* **Combinatorial Fold Optimization:** [`shealert_audio_pipeline/analysis/find_balanced_split.py`](file:///f:/SheAlert-Women-Safety-System/shealert_audio_pipeline/analysis/find_balanced_split.py)
* **Automated Unit & Integration Tests:** [`shealert_audio_pipeline/tests/test_manifest.py`](file:///f:/SheAlert-Women-Safety-System/shealert_audio_pipeline/tests/test_manifest.py)
* **Verified Split Summary:** [`shealert_audio_pipeline/output/corpus_summary.txt`](file:///f:/SheAlert-Women-Safety-System/shealert_audio_pipeline/output/corpus_summary.txt)
