# SheAlert: Project Context File

> Purpose: single source of truth for coding assistance in the IDE. Converted from the FYP proposal "SheAlert: An Automatic Women Safety System Using Sensors, Audio, and Camera" (Version 2.1, Dataset Expanded).
> Institution: Air University, Islamabad, Department of Creative Technologies. Program: BS Artificial Intelligence.
> Requirement IDs (OB-n, FR-m.n, LM-n) are kept exactly as in the proposal so code, tests, and commit messages can reference them.

---

## 0. How to use this file

- Treat the FR-x.y items as the functional spec. Reference them in code comments and tests (example: `# FR-1.2 sliding window`).
- Values marked **fixed** are hard numbers from the spec. Values marked **TBD** are intentionally not pre-committed and must be decided from experiments (F1 targets, battery figures, fusion threshold, fusion weights).
- Section 17 lists inconsistencies inside the source document so you do not follow stale details by mistake.

---

## 1. One-paragraph summary

SheAlert is a fully passive Android app that watches motion (accelerometer + gyroscope) and audio (microphone) in the background, fuses both into a live threat score, and automatically sends an SOS (GPS + 5 s audio clip + optional camera snapshot) to pre-registered contacts. Before sending, a two-stage cancellation window (10 s silent + 10 s audible, with a dead man's switch PIN) gives the user a chance to cancel a false alarm. All AI inference runs on-device. It targets Urdu-speaking users in Pakistan, using a YAMNet model fine-tuned on a purpose-built Urdu assault-distress corpus.

**Core idea:** existing apps need a button press or shake. A person being grabbed or frozen by fear often cannot do that, so the phone must decide by itself.

---

## 2. Team and ownership

| Student | Reg. No. | Role | Modules |
|---|---|---|---|
| Mahad Jokhio | 231242 | Motion and Sensor AI | Module 1, Module 2 |
| Moaz Hassan Khan Manj | 231168 | Audio + Fusion (core AI logic) | Module 3, Module 4, Module 5 |
| Hunain Ahmed | 231156 | System intelligence and validation | Module 6, 7, 8, 9 + integration + ablation study |

- Supervisor: Dr. Samana Batool (Assistant Professor)
- Co-Supervisor: Dr. Madiha Yousaf (Assistant Professor)
- Industrial Supervisor: Maryam Aamir Ashraf (Associate Data Scientist)
- Shared by all three: dataset recording, labelling, validation, testing (unit, integration, end-to-end), final integration support, documentation and report writing.

### Per-student detail

**Mahad (Modules 1 and 2):** UCI HAR, WISDM, MobiAct, UniMiB SHAR preparation; sliding-window segmentation and normalisation; Android SensorManager continuous collection in Kotlin; multiclass LSTM design, training, evaluation; four-class corpus assembly; volunteer recording for Struggle/Resistance class; ablation vs LSTM Autoencoder baseline; TFLite conversion and deployment; sensor-only fallback; threat score stream delivered to Student 3; original experiment: placement variability study (pocket vs bag vs hand).

**Moaz (Modules 3, 4, 5):** noise cancellation stage (FR-3.0); Foreground Service; WebRTC VAD (Tier 1); rule-based filter (Tier 2); microphone interruption handler; battery benchmarking (Month 1 spike); integration of UrduSER, UrduSpeech, SEMOUR, CREMA-D, RAVDESS and ambient datasets; emotion-field filtering; original Urdu recording coordination; labelling and inter-annotator agreement; two-stage YAMNet fine-tuning; 5-fold CV and OOD evaluation; TFLite conversion; public dataset release; late-fusion formula; context-adaptive weights; threshold tuning and ROC analysis.

**Hunain (Modules 6 to 9):** confirmation window UI and dead man's switch PIN; camera module (MobileNet SSD, fail-safe); SOS and alert delivery (GPS packaging, Firebase FCM, SMS fallback, FastAPI backend, acknowledgement timer); Flutter app (onboarding, modes, permission monitor, feedback, heatmap); full system integration; ablation study (formal owner).

---

## 3. Problem statement (short form)

1. **Manual-action assumption.** Existing apps need a deliberate user action. Trauma literature on tonic immobility and peritraumatic freezing shows many victims cannot act. Attackers may also take the phone. Continuous GPS sharing tells contacts where the user is, not whether something is wrong.
2. **Language and cultural gap.** Published audio-safety work is English-centric (mostly AudioSet). Urdu distress phrases (bachao, madad karo, choro mujhe) are not in any indexed safety dataset. Pakistan population is about 241 million.
3. **Battery and deployment constraint.** Always-on audio inference can burn more than 15% battery per hour on mid-range hardware without optimisation. Elevated drain is an anticipated risk (not confirmed). Mitigation: 3-tier cascade plus noise cancellation. Real consumption is measured in Month 1.

These three problems reinforce each other, so the system must solve them together on a standard Android phone with no extra hardware.

---

## 4. System architecture

### 4.1 End-to-end data flow

```
                        (only when OUTSIDE home geofence, FR-3.6)

 Accelerometer + Gyro (50 Hz)                 Microphone (16 kHz mono)
        |                                              |
 [M1] 2 s window, 50% overlap,                 [FR-3.0] Noise cancellation
      z-score normalise                        (spectral subtraction + Wiener)
        |                                              |
 [M2] Multiclass LSTM (TFLite)                 [FR-3.2] Tier 1: WebRTC VAD + RMS
   4-class softmax                                      | voice present
        |                                      [FR-3.3] Tier 2: rule-based filter
   s = max non-Normal prob                              | 2 of 3 rules pass
        |                                      [FR-3.4] Tier 3: YAMNet (TFLite)
        |                                                |
        |                                          a = audio confidence
        +---------------------+--------------------------+
                              |
                   [M5] Late fusion: threat = W1*s + W2*a   (W1 + W2 = 1)
                        context-adaptive weights, ROC-chosen threshold
                              |  threat >= threshold
                   [M6] Stage 1: 10 s SILENT window (cancel or PIN)
                              |  not cancelled
                        Stage 2: 10 s AUDIBLE window (beep, flash)
                              |  not cancelled
                   [M7] Optional camera check, 2 to 3 s, MobileNet SSD
                        (fail-safe: SOS proceeds regardless)
                              |
                   [M8] SOS: GPS + 5 s audio clip + optional snapshot
                        FCM primary, SMS fallback after 60 s no ack
                              |
                   [M9] Flutter UI, heatmap, post-event feedback
                        (feedback feeds per-user threshold, FR-5.4)
```

### 4.2 Key design principles

- Fully passive: no user action required during a crisis.
- All inference on-device. No cloud AI (LM-5). Backend (FastAPI) only relays alerts and contacts.
- Late fusion chosen because modalities can fail independently (mic blocked, sensors still working).
- Fail-safe defaults: camera failure does not block SOS (FR-7.3); location fix failure defaults to active monitoring (LM-8); SOS cannot start with zero contacts (FR-8.5).
- Cascaded early-exit audio pipeline to save battery.

### 4.3 Interfaces between modules (derived from the spec)

| From | To | Payload |
|---|---|---|
| M1 | M2 | Normalised tensor per 2 s window (accel + gyro) |
| M2 | M5 | Softmax class-probability vector, threat score 0 to 1 (FR-2.2) |
| M1 / M2 | M5 | Hardware failure flag (FR-1.5), sensor-only fallback status (FR-2.5) |
| M3 / M4 | M5 | Audio confidence score `a`; mic interruption flag (FR-3.5) |
| M5 | M6 | Trigger when fused threat crosses threshold |
| M6 | M7 | Trigger after both confirmation stages pass |
| M6 / M7 | M8 | Confirmed SOS event |
| M9 | M5 | Post-event true/false feedback for threshold personalisation |
| M9 | M3 | Home geofence state (mic on outside, paused inside) |

---

## 5. Objectives

**OB-1 (Motion classifier).** Build and evaluate a supervised multiclass LSTM for passive motion threat detection. Normal and Fall classes from UCI HAR, WISDM, MobiAct, UniMiB SHAR. Struggle/Resistance and Forced Movement from original volunteer recordings. Metric: per-class F1 on a subject-disjoint test set over four classes. Targets set after implementation (no pre-committed threshold). LSTM Autoencoder kept only as an ablation baseline.

**OB-2 (Urdu audio corpus and YAMNet).** Build a multi-source Urdu distress corpus (see Section 9). Apply noise cancellation to all audio before feature extraction. Two-stage YAMNet fine-tuning (Stage 1 on full combined corpus, Stage 2 on original Urdu recordings only). Evaluate per-class F1 via 5-fold speaker-stratified cross-validation (mean F1 +/- std) plus out-of-distribution evaluation on one held-out acoustic environment. Report the generalisation gap. Targets defined after CV results.

**OB-3 (Fusion).** Context-adaptive weighted fusion engine. Operating threshold chosen via ROC analysis on a held-out set. Full ROC figures in thesis.

**OB-4 (Confirmation window).** Two-stage cancellation window with dead man's switch PIN. **Target: at least 60% relative reduction in false-alarm SOS transmissions**, comparing full system (with window) vs same pipeline without window, on the identical held-out set (ablation condition (c) vs (d), FR-5.5).

**OB-5 (Battery benchmark).** Benchmark three audio pipeline configurations on three device tiers (budget, mid-range, flagship) using Android Battery Historian:
- (a) always-on full YAMNet
- (b) two-tier cascade (VAD + YAMNet)
- (c) three-tier cascade (VAD + noise cancellation + rule-based filter + YAMNet)

Production configuration is chosen from measured results. No consumption figure is pre-committed.

**OB-6 (Integration).** Integrate all modules in a Flutter Android app. Evaluate end-to-end detection latency, false-positive rate, battery consumption.

**OB-7 (Ablation).** Show multi-modal fusion beats any single-modality baseline in both detection rate and false-positive rate.

---

## 6. Module specifications

### Module 1: Motion Sensor Data Pipeline (owner: Mahad)

| ID | Requirement |
|---|---|
| FR-1.1 | Continuous accelerometer and gyroscope collection via Android `SensorManager` and `SensorEventListener` at **50 Hz** |
| FR-1.2 | Real-time sliding-window segmentation: **2 s window, 50% overlap** (100 samples per window, hop of 1 s) |
| FR-1.3 | Per-window z-score normalisation using mean and std **pre-computed during model training** |
| FR-1.4 | Normalised tensor streamed to Module 2, end-to-end pipeline latency **below 20 ms per window** |
| FR-1.5 | Sensor hardware availability monitor; raises a flag to the fusion engine on hardware failure or API disconnection |

### Module 2: Motion Threat Classification, Multiclass LSTM (owner: Mahad)

| ID | Requirement |
|---|---|
| FR-2.1 | Supervised multiclass LSTM with four classes (below) |
| FR-2.2 | Per-window softmax class-probability vector streamed to fusion engine as threat score on 0 to 1 scale |
| FR-2.3 | **Subject-disjoint** train/test split mandatory (zero subject overlap). 5-fold CV with subject-level stratification. Report mean F1 +/- std. No pre-committed threshold |
| FR-2.4 | TensorFlow Lite on-device inference, target latency **below 30 ms per window** on a mid-range Android device |
| FR-2.5 | Sensor-only fallback mode delivering continuous class scores when audio module is interrupted or unavailable |
| FR-2.6 | Placement variability study (original experiment): accuracy across pocket, bag, and hand placements |
| FR-2.7 | LSTM Autoencoder [14] kept as baseline in ablation only; compare per-class F1 of multiclass LSTM vs autoencoder anomaly-detection rate to quantify false-positive reduction |
| FR-2.8 | Volunteer collection for Struggle/Resistance using Physics Toolbox Sensor Suite (Android, CSV export); written informed consent; 10 to 15 min per volunteer |

**Four motion classes (FR-2.1):**

| Class | Source |
|---|---|
| (a) Normal Activity: walking, running, sitting, standing, stairs | UCI HAR [1], WISDM [12] |
| (b) Fall/Collapse | MobiAct [23] (four fall types, 66 subjects) and UniMiB SHAR [24] (eight fall types, 30 subjects); both Android smartphone |
| (c) Struggle/Resistance | Original volunteer recordings (wrist grab, forced directional pull, push/stumble); 15 to 20 campus volunteers; no public dataset exists |
| (d) Forced Movement | Subset of MobiAct dynamic ADL classes plus volunteer recordings of walking under physical constraint |

### Module 3: Audio Pipeline and 3-Tier VAD Cascade (owner: Moaz)

| ID | Requirement |
|---|---|
| FR-3.0 | **Noise cancellation pre-processing** applied before all tiers: spectral subtraction + Wiener filter to suppress stationary noise (traffic, HVAC, crowd chatter). Purposes: cleaner features for YAMNet, fewer spurious Tier 2 activations (battery). CPU only, O(N log N) per frame |
| FR-3.1 | Android Foreground Service with `foregroundServiceType=microphone` (required on Android 14+) |
| FR-3.2 | **Tier 1:** WebRTC VAD + RMS energy threshold on the noise-cancelled signal. Chunks without voice are discarded immediately |
| FR-3.3 | **Tier 2:** rule-based filter on noise-cancelled signal: RMS amplitude **> 8,000**, fundamental pitch **> 300 Hz**, zero-crossing rate **> 0.3**. **At least 2 of 3** must match |
| FR-3.4 | **Tier 3:** YAMNet inference [8, 9] only when Tier 2 is satisfied. Early-exit design [13]. Battery benefit is anticipated, to be confirmed by Month 1 benchmark (OB-5) |
| FR-3.5 | Mic interruption detection: raise flag to fusion engine if audio lost for **>= 5 consecutive seconds** |
| FR-3.6 | Mic active **only outside home geofence**. Pipeline starts when GPS is outside the user-set home radius and pauses on return. Location via `FusedLocationProviderClient` at low-power priority |

### Module 4: Urdu Distress Dataset and YAMNet Classification (owner: Moaz)

| ID | Requirement |
|---|---|
| FR-4.0 | External public corpus integration (Section 9 has details) |
| FR-4.1 | Original recording of 800 to 1,000 Urdu clips (Section 10 has protocol). All passed through FR-3.0. Augmentation on original clips only, for regularisation, not a substitute for diversity |
| FR-4.2 | Combined pre-augmentation corpus about 14,000 to 16,000 clips; about 16,000 to 19,000 training items after 2 to 3x augmentation of original clips only |
| FR-4.3 | MFCC and Mel-spectrogram features via a Librosa-equivalent on-device pipeline, applied after noise cancellation |
| FR-4.4 | Two-stage YAMNet fine-tuning [8]. **Stage 1:** full combined corpus (UrduSER + UrduSpeech + SEMOUR + CREMA-D + RAVDESS + AudioSet + ESC-50 + UrbanSound8K + original). **Stage 2:** original Urdu recordings only, to specialise on bachao / madad karo / choro mujhe and Pakistani urban acoustics. Evaluate both stages separately and report an incremental comparison table. Per-class F1 via 5-fold speaker-stratified CV plus OOD on one held-out acoustic environment |
| FR-4.5 | TFLite conversion and on-device deployment. Release original Urdu assault-distress dataset publicly under **CC-BY-4.0** at project end (SEMOUR clips excluded from release, see LM-11) |

**Audio classes:** Distress, Aggression, Normal.

### Module 5: Multi-Modal Fusion Engine (owner: Moaz)

| ID | Requirement |
|---|---|
| FR-5.1 | Late fusion: `threat = W1*s + W2*a`, with `W1 + W2 = 1`. `s` = highest non-Normal class probability from the LSTM. `a` = audio confidence score |
| FR-5.2 | Context-adaptive weights: increase W1 in noisy environments (audio less reliable), increase W2 in quiet environments. Weights tuned empirically |
| FR-5.3 | Threshold chosen by ROC-curve analysis [7] on a held-out set. Not preset |
| FR-5.4 | Per-user threshold personalisation: after **two weeks** of post-event feedback, adjust threshold based on false-alarm history |
| FR-5.5 | Ablation protocol (formal owner: Hunain). See Section 12 |

### Module 6: Two-Stage Confirmation Window (owner: Hunain)

| ID | Requirement |
|---|---|
| FR-6.1 | **Stage 1 (silent):** 10 s countdown, minimum brightness, soft vibration, large CANCEL button |
| FR-6.2 | **Stage 2 (audible):** 10 s countdown, loud beep, screen flash, CANCEL button. Only runs if Stage 1 passes |
| FR-6.3 | **Dead man's switch PIN:** 4-digit PIN required to confirm safety. An attacker cannot cancel without it |
| FR-6.4 | Evaluated as a human-in-the-loop false-alarm reduction system; contribution reported in ablation |
| FR-6.5 | Usability evaluation under simulated stress with 10 to 15 volunteers |

### Module 7: Camera Module (owner: Hunain)

| ID | Requirement |
|---|---|
| FR-7.1 | Optional camera confirmation using MobileNet SSD [17] for **2 to 3 s** after both confirmation stages pass |
| FR-7.2 | Evaluated as third-modality contribution: compare two-modality (sensor + audio) vs three-modality precision |
| FR-7.3 | **Fail-safe:** SOS proceeds regardless of camera availability or hardware failure |
| FR-7.4 | On-device TFLite inference, target **below 500 ms** on a mid-range Android device |

### Module 8: SOS and Alert Delivery (owner: Hunain)

| ID | Requirement |
|---|---|
| FR-8.1 | Package: GPS from `FusedLocationProviderClient`, **5 s audio clip** from rolling buffer, optional camera snapshot |
| FR-8.2 | Primary delivery via Firebase Cloud Messaging; each contact gets a one-tap acknowledgement |
| FR-8.3 | Automatic SMS fallback via `SmsManager` if no FCM acknowledgement within **60 s** |
| FR-8.4 | FastAPI (Python) backend for alert forwarding and contact relay; JWT-authenticated admin endpoints |
| FR-8.5 | Registration guard: SOS cannot be initiated if no contacts are registered |

### Module 9: Flutter App, UI, Heatmap, Feedback (owner: Hunain)

| ID | Requirement |
|---|---|
| FR-9.1 | Mandatory onboarding: at least 1 emergency contact, OEM battery-optimiser whitelist prompts (MIUI, One UI, ColorOS, EMUI), Android 12+ microphone-indicator explanation |
| FR-9.2 | Dashboard: real-time threat-score gauge, monitoring status, sensitivity mode, contacts list |
| FR-9.3 | Sensitivity modes: **Safe** (raised threshold), **Standard** (validated default), **Aggressive** (lowered threshold) |
| FR-9.4 | Persistent permission health monitor: check critical permissions on every foreground event; persistent red notification if any missing |
| FR-9.5 | Post-event feedback: user labels each alert true or false; feeds FR-5.4 |
| FR-9.6 | Anonymous GPS-timestamped incident heatmap; no personal data, names, or device identifiers stored |
| FR-9.7 | Home location: live fix or manual pin, geofence radius **50 to 500 m, default 100 m**. Dashboard shows "Monitoring On" outside, "Paused - At Home" inside |

---

## 7. Quick-reference constants (for code)

| Parameter | Value | Ref |
|---|---|---|
| Sensor sampling rate | 50 Hz | FR-1.1 |
| Window length / overlap | 2 s / 50% | FR-1.2 |
| Motion pipeline latency | < 20 ms per window | FR-1.4 |
| LSTM inference latency | < 30 ms per window (mid-range) | FR-2.4 |
| Audio sample format | 16 kHz, 16-bit, mono | Sec. 10 |
| Tier 2 RMS | > 8,000 | FR-3.3 |
| Tier 2 pitch | > 300 Hz | FR-3.3 |
| Tier 2 ZCR | > 0.3 | FR-3.3 |
| Tier 2 rule | at least 2 of 3 | FR-3.3 |
| Mic loss flag | >= 5 s | FR-3.5 |
| Stage 1 / Stage 2 window | 10 s / 10 s | FR-6.1, 6.2 |
| PIN length | 4 digits | FR-6.3 |
| Camera check | 2 to 3 s, < 500 ms inference | FR-7.1, 7.4 |
| SOS audio clip | 5 s | FR-8.1 |
| SMS fallback delay | 60 s without FCM ack | FR-8.3 |
| Geofence radius | 50 to 500 m, default 100 m | FR-9.7 |
| Geofence detection lag | up to 60 s in poor GPS | LM-8 |
| Threshold personalisation | after 2 weeks of feedback | FR-5.4 |
| Fusion weights | W1 + W2 = 1, TBD | FR-5.1 |
| Fusion threshold | TBD (ROC) | FR-5.3 |
| Target FA reduction from window | >= 60% relative | OB-4 |
| Foreground service type | `microphone` (Android 14+) | FR-3.1 |
| Min Android API for service | API 29+ (Android 10) | Sec. 11 |
| Tested Android versions | 10, 12, 14 | Sec. 7 spike |

---

## 8. Motion data plan (Module 2)

- Public: UCI HAR [1], WISDM [12], MobiAct [23], UniMiB SHAR [24]. Direct downloads.
- Original (Struggle/Resistance and Forced Movement), see Section 10.1.
- Contingency: if volunteers fall below 15 by end of Month 2, augmentation multiplier goes from 3x to 5x and the generalisability claim is narrowed in the thesis.

---

## 9. Audio data plan (Module 4)

### 9.1 Class mapping

| Source | Clips (relevant) | License | Mapping |
|---|---|---|---|
| UrduSER [20] | 3,500 total; about 1,500 used (about 500 each) | CC BY 4.0 | Fear -> Distress, Angry -> Aggression, Neutral -> Normal. 10 professional Pakistani actors, drama-serial sourced |
| UrduSpeech [21] (Attia) | about 2,000 to 3,000 after filtering | CC BY 4.0 | US-Std split (59.2 h of 156 h corpus). Filter by emotion field: fearful/anxious -> Distress, angry -> Aggression, podcast/vlog/comedy neutral -> Normal |
| SEMOUR [22] | 15,040 total; about 5,640 used (about 1,880 each) | CC non-commercial research | Fearful -> Distress, Anger -> Aggression, Neutral -> Normal. 8 actors, studio quality |
| CREMA-D [5] | 7,442 total; about 2,400 Fear + Anger | CC BY-SA 4.0 | Cross-lingual acoustic anchor for Stage 1 |
| RAVDESS (audio-only) | about 200 angry/fearful | CC BY-NC 4.0 | Cross-lingual anchor |
| AudioSet scream subset | about 600 to 800 (about 700) | n/a | Supplements Distress |
| ESC-50 + UrbanSound8K | about 800 | n/a | Supplements Normal (urban ambient) |
| Original Urdu recordings | 800 to 1,000 | CC-BY-4.0 on release | All three classes; Stage 2 fine-tuning |

Total pre-augmentation: about **14,000 to 16,000** clips. After augmentation of originals only: about 16,000 to 19,000.

Keep all three Urdu corpora in full before emotion-field filtering (preserves reclassification flexibility).

### 9.2 Why each source exists

- UrduSER: broadcast naturalistic register, validated by 100 native speakers (94% inter-rater accuracy).
- UrduSpeech: scale and multi-domain naturalistic coverage.
- SEMOUR: controlled studio, phonetically balanced.
- CREMA-D / RAVDESS: language-independent fear/anger acoustics (pitch spikes, energy bursts, vocal tremor) before any Urdu data.
- Stage 2 original clips: assault-specific vocabulary and Pakistani urban acoustics.

### 9.3 Augmentation policy

Noise addition, pitch shift, time stretch. Applied to original clips only, as regularisation. It must not be presented as a source of speaker or environment diversity.

---

## 10. Data gathering protocols

### 10.1 Motion volunteers (Struggle/Resistance, Forced Movement)

- Volunteers: 15 to 20; at least 50% female; age 18 to 45; at least 3 regional accent backgrounds (Punjabi, Pashtun, standard Urdu); no more than 3 from the same immediate social group. Recruited across 3 weeks of Month 2 via notice board and direct approach.
- Equipment: mid-range Android phone in trouser pocket; Physics Toolbox Sensor Suite (CSV); accel + gyro at 50 Hz; open campus area or corridor with safety mat.
- Per volunteer (10 to 15 min total):
  1. Scenario 1, wrist grab: second person grabs wrist from behind while walking; subject resists 5 to 10 s.
  2. Scenario 2, forced directional pull: grab arm and pull in another direction against resistance.
  3. Scenario 3, push/stumble: push from side; subject stumbles without falling.
  4. Each scenario repeated 3 times; recording window 10 to 15 s per repetition.

### 10.2 Audio volunteers

- Volunteers: 20 to 30; at least 60% female; age 18 to 45; at least 4 regional accent backgrounds (Punjabi, Sindhi, Pashtun, standard Urdu); max 3 from same social group. Recruited in Weeks 1 to 2 of Month 2.
- Six real environments (no artificial substitution): (1) quiet indoor classroom after hours, (2) indoor ambient cafeteria, (3) outdoor quiet campus lawn, (4) outdoor busy main gate / G-9 markaz, (5) indoor reverberant corridor, (6) vehicle interior (car or rickshaw). Each volunteer records in at least 2.
- Device: mid-range Android at 30 cm from mouth; Android `AudioRecord` at 16 kHz, 16-bit mono.
- Per volunteer (15 to 20 min): Distress 15 clips (bachao x5, madad karo x5, choro mujhe x5), Aggression 5 clips (threatening Urdu speech), Normal 5 clips (casual Urdu). Total 25 clips of about 5 s.

### 10.3 Labelling

- Two annotators label independently; third team member breaks ties.
- Cohen's Kappa target **>= 0.80**, reported in thesis.

### 10.4 Quality control

- Reject clips shorter than 3 s.
- Reject clips with amplitude clipping.
- Minimum SNR 10 dB.
- No class above 40% of the original Urdu corpus before augmentation.

### 10.5 Ethics

- Signed written informed consent before recording.
- Supervisor sign-off as IRB-equivalent per Air University policy.
- Anonymous numeric ID per volunteer; no personal identifiers with audio files.
- Right to withdraw and have all clips deleted at any time.

### 10.6 User research and system evaluation

- Interviews and questionnaire with 5 to 8 women from the target group (validate confirmation window, sensitivity modes, notification format, onboarding UX).
- End-to-end evaluation with 10 to 15 campus volunteers: simulated daily activities and two staged threat scenarios. Measure real-world false-positive rate, detection latency, confirmation-window usability.

---

## 11. Tech stack

| Area | Tool | Version / note | Purpose |
|---|---|---|---|
| Language | Python | 3.10+ | Model development, training, augmentation |
| ML framework | PyTorch / TensorFlow | 2.x | Multiclass LSTM, YAMNet fine-tuning |
| Audio features | Librosa | 0.10+ | MFCC, Mel-spectrogram |
| Noise cancellation | SciPy / WebRTC NS | latest | Spectral subtraction, Wiener filter (FR-3.0) |
| Audio base model | YAMNet | TF Hub | Pretrained base for distress classifier |
| Sensor model | Multiclass LSTM | custom | Four-class motion threat detection |
| On-device inference | TensorFlow Lite | 2.x | Quantised deployment |
| Mobile framework | Flutter (Dart) | 3.x | App and UI |
| Native Android | Kotlin | API 29+ | Foreground service, sensors, background mic |
| Background CPU | `PARTIAL_WAKE_LOCK` | API 1+ | Keep CPU running when screen locked |
| Tier 1 filter | WebRTC VAD | latest | Voice activity detection |
| Vision | MobileNet SSD | latest | Person detection |
| Primary alert | Firebase Cloud Messaging | latest | SOS push |
| Fallback alert | Android `SmsManager` | API 1+ | SMS fallback |
| Location | `FusedLocationProviderClient` | latest | GPS + Wi-Fi + cell fusion |
| Backend | FastAPI (Python) | 0.100+ | Alert forwarding, contacts, JWT admin |
| Battery benchmark | Android Battery Historian | latest | Measure pipeline configs |
| Battery optimisation | Battery Optimisations API | API 23+ | OEM whitelist prompt |
| Evaluation | scikit-learn | 1.x | ROC, F1, precision, recall, Cohen's Kappa |
| Sensor recording | Physics Toolbox Sensor Suite | latest | Volunteer IMU data |
| IDE | VS Code + Android Studio | latest | Python, Dart, Kotlin |

**Why YAMNet (2019) and not AST/BEATs:** lightweight, mature TFLite tooling, strong AudioSet benchmarks. Transformer audio models are more accurate but exceed on-device latency and battery budget on mid-range Android.

---

## 12. Evaluation plan and ablation study (FR-5.5, owner: Hunain)

Identical held-out test set with both simulated threat scenarios and normal-life recordings.

| Condition | Description |
|---|---|
| (a) | Sensor-only (multiclass LSTM) |
| (b) | Audio-only (YAMNet) |
| (c) | Sensor + audio fusion, no confirmation window |
| (d) | Full system with confirmation window |
| (e) | LSTM Autoencoder baseline vs multiclass LSTM (false-positive reduction) |
| (f) | Pipeline without noise cancellation vs with noise cancellation (FR-3.0 contribution) |

Metrics: per-class F1, false-alarm rate, end-to-end latency. OB-4's 60% target is the relative reduction in false-alarm SOS transmissions between (c) and (d).

Other evaluations:
- Motion: subject-disjoint test, 5-fold CV, mean F1 +/- std.
- Audio: 5-fold speaker-stratified CV, OOD on one held-out acoustic environment, generalisation gap reported, Stage 1 vs Stage 2 incremental table.
- Fusion: ROC curves and chosen operating point.
- Battery: three configs x three device tiers (Battery Historian).
- Camera: two-modality vs three-modality precision.
- Placement variability (pocket, bag, hand).
- Usability under simulated stress (10 to 15 volunteers).

---

## 13. Limitations and constraints

| ID | Summary |
|---|---|
| LM-1 | Android-only. iOS excluded due to background mic and sensor restrictions |
| LM-2 | Background mic behaviour varies across OEM firmware (MIUI, One UI, ColorOS, EMUI). Month 1 spike documents workarounds |
| LM-3 | Struggle class is simulated by campus volunteers; generalisation to real assaults is a known limitation. If audio/motion recruitment falls below 15 by end of Month 2, augmentation goes 3x to 5x and claims narrow |
| LM-4 | Camera is optional; unavailability never blocks SOS |
| LM-5 | All inference on-device; no cloud AI; performance depends on device CPU/GPU |
| LM-6 | Research prototype; no clinical or legal certification |
| LM-7 | Indoor and dense-urban GPS accuracy is limited; Fused provider mitigates |
| LM-8 | Geofence detection may lag up to 60 s in poor GPS. On sustained fix failure, default to active monitoring |
| LM-9 | Battery drain anticipated to be elevated (CPU, 50 Hz polling, noise cancellation, periodic inference). Not a confirmed outcome. No figure pre-committed. Mitigated by cascade plus FR-3.0. Month 1 benchmark gates architecture |
| LM-10 | Noise cancellation is tuned for stationary noise; may degrade in dynamic environments. Assessed in OOD evaluation |
| LM-11 | SEMOUR is non-commercial research only. Compatible with this FYP and the CC-BY-4.0 release because SEMOUR clips are not included in the release. Document compliance in thesis |

### Explicit scope exclusions

iOS, web interfaces, server-side AI processing, real-time video streaming, persistent local audio or video storage during normal monitoring, clinical or legal certification.

---

## 14. Risks and contingencies

| Risk | Plan |
|---|---|
| Volunteer recruitment below 15 by end of Month 2 (Struggle class) | Raise augmentation from 3x to 5x; narrow generalisability claim to recorded demographics |
| Camera (Module 7) hits persistent OEM compatibility issues | Demote to optional appendix contribution; two-modality (sensor + audio) system becomes the primary evaluated system |
| Recording delays | All public datasets pre-downloaded in Month 1 as buffer |
| Background mic blocked on some OEMs / Android versions | Month 1 technical spike on Android 10, 12, 14 across three device tiers; OEM whitelist prompts in onboarding |
| Battery too high | Cascade + noise cancellation; select production config from Month 1 measurements |

Core deliverables remain completable even if both contingencies trigger: multiclass LSTM, fine-tuned YAMNet, fusion engine, confirmation window, SOS delivery.

**Mandatory Month 1 spike:** validate continuous background mic access on Android 10, 12, 14 across three device tiers, and benchmark battery for three pipeline configurations before architecture is finalised.

---

## 15. Timeline (12 months)

**Part 1 (Months 1 to 6): research, datasets, model development**

| Task | Owner | Months |
|---|---|---|
| Literature review and gap analysis | All | M1 |
| Architecture finalisation | All | M1 |
| Tool setup and environment | All | M1 |
| Mic spike: Android 10/12/14 (milestone) | Moaz | M1 |
| UCI HAR and WISDM preprocessing | Mahad | M2 to M3 |
| Urdu audio recording | All | M2 to M3 |
| Augmentation pipeline | Moaz | M3 |
| Volunteer anomaly test recordings | All | M3 |
| LSTM training and evaluation | Mahad | M4 to M6 |
| YAMNet fine-tuning and evaluation | Moaz | M4 to M6 |
| Module evaluation done (milestone) | All | M6 |
| TFLite model conversion | All | M7 |
| Android sensor pipeline (Kotlin) | Mahad | M7 to M8 |
| 3-tier audio cascade on Android | Moaz | M7 to M8 |
| Score streams delivered (milestone) | All | M8 |
| Fusion engine and adaptive weights | Moaz | M9 |
| ROC-curve threshold validation | Moaz | M9 |

**Part 2 (Months 7 to 12): integration, evaluation, submission**

| Task | Owner | Months |
|---|---|---|
| Two-stage confirm window + PIN | Hunain | M9 |
| Camera module (MobileNet SSD) | Hunain | M9 |
| All modules into Flutter app | Hunain | M10 |
| SOS system and Firebase backend | Hunain | M10 |
| Feedback loop, heatmap, onboarding | Hunain | M10 |
| End-to-end smoke testing | All | M10 |
| End-to-end system testing | All | M11 |
| Ablation study (4 conditions) | Hunain | M11 |
| Battery benchmarking (3 device tiers) | All | M11 |
| Placement variability study | Mahad | M11 |
| User scenario testing (10 to 15 volunteers) | All | M11 |
| Individual thesis writing (x3) | All | M12 |
| Group project report and docs | All | M12 |
| Final demo preparation and rehearsal | All | M12 |
| Plagiarism report and final defence (milestone) | All | M12 |

Note: the Gantt in the source labels the LSTM task as "LSTM Autoencoder training and eval". This is stale; see Section 17.

---

## 16. UI mockups (six core screens)

Dark theme, teal/green accents for safe state, amber for Stage 1, red for Stage 2 and SOS. Bottom navigation: Home, Status, Map, Settings. Login, Signup, Forgot Password, About are excluded from mockups per FYP template.

1. **Main Dashboard.** Header "SheAlert - Passive monitoring active" with LIVE badge. Semi-circular **Threat Score gauge** (example 0.18, "Environment Safe"). **Module Status** list: Motion Sensor (active, score), Audio Monitor (Tier 1 active), GPS Location (locked), each with a status dot. **Emergency Contacts** list with READY badges (example: Ammi Jaan / Mother, Sara Baji / Sister).
2. **Stage 1 Silent Confirmation.** Amber. "Possible Threat Detected. Stay calm. If safe, press Cancel. Alert fires in 10 seconds." Circular countdown ring (example: 7 s). Notice "Silent vibration active, screen at minimum brightness". Large "I AM SAFE - CANCEL" button. Hint: enter safety PIN instead if PIN mode enabled.
3. **Stage 2 Audible Confirmation.** Red. "SOS About to Fire. Alarm active. Alert sends in 10 seconds unless cancelled." Countdown ring (example: 4 s). Notice "Alarm tone sounding, GPS + audio clip being prepared". "CANCEL - I AM SAFE" button. "Why cancelling?" chips: False alarm, Pressed by mistake, Test mode.
4. **SOS Dispatched.** "Alert Dispatched, contacts notified with location and evidence", sent timestamp. **Contacts Notified** with Delivered status and an "I'm Coming" acknowledgement from a contact. **Evidence Sent:** GPS location, audio clip (5-second recording), camera snapshot (1 image). "Mark Myself as Safe" button.
5. **Incident Heatmap.** "Islamabad, last 30 days", report count badge (example 23 REPORTS). Blurred heat blobs on a grid with area labels. Legend High / Med / Low, "Anonymous only". Note: "GPS and timestamps only, no personal data stored".
6. **Settings.** **Detection Mode** selector: Safe / Standard / Aggressive. **Module Controls** toggles: Motion Sensor (Accelerometer + Gyro), Audio Monitor (3-tier cascade pipeline), PIN Cancel Mode (Require PIN to cancel, default off). **Emergency Contacts** list with remove buttons and "+ Add Contact".

Also required by spec but not in the mockups: onboarding flow, home-location and geofence setting (FR-9.7), permission health notification (FR-9.4), post-event feedback (FR-9.5).

---

## 17. Notes and inconsistencies in the source (read before coding)

1. **Stale WBS and Gantt labels.** The WBS figure and Gantt (unchanged from v1.0) still say "Motion Anomaly Detection (LSTM AE)" and "LSTM Autoencoder training and eval", and the WBS mentions "Urdu/Punjabi". The authoritative design (Sections 4, 8.2, OB-1) is the **supervised multiclass LSTM**. The autoencoder is an ablation baseline only.
2. **Augmentation multiplier.** FR-4.2 says 2 to 3x augmentation on original clips; contingency text raises 3x to 5x for the Struggle class. Treat 3x as the default and 5x as the contingency.
3. **Volunteer counts differ by modality.** Motion: 15 to 20 volunteers. Audio: 20 to 30 volunteers. The below-15 contingency is stated in both places.
4. **Dashboard mockup vs spec.** The mockup dashboard lacks explicit sensitivity-mode display and "Monitoring On / Paused - At Home" text required by FR-9.2 and FR-9.7. Implement per the FR text.
5. **Mockup timestamps** (14:32 vs 14:35 for SENT) are placeholders, not spec.
6. **Tier 2 RMS > 8,000** assumes 16-bit PCM amplitude scale. Confirm the scale used after noise cancellation when implementing.
7. **Threat score `s`** for fusion is the highest non-Normal class probability, not the raw softmax vector (FR-5.1 vs FR-2.2).
8. **UrduSpeech reference** (arXiv:2605.17846, 2026) and dataset counts are as stated in the proposal; verify class filtering when downloading.
9. **Placeholder targets.** F1 targets, battery figures, fusion threshold, and fusion weights are intentionally undefined. Do not hard-code guessed values as requirements; expose them as config.

---

## 18. Panel comment history (Version 1.0 to 2.1)

| Comment | Concern | Resolution |
|---|---|---|
| 1 | LSTM autoencoder on normal activity gives many false positives; use multiclass | Replaced with supervised multiclass LSTM (four classes); autoencoder kept as ablation baseline (FR-2.7); volunteer protocol for Struggle class (FR-2.8) |
| 2 | Dataset too small; augmentation cannot add diversity | Added UrduSER (v2.0), UrduSpeech, SEMOUR, CREMA-D (v2.1), AudioSet, RAVDESS, ESC-50, UrbanSound8K; raised original recording target from 450+ clips / 8 to 10 volunteers / 4 environments to 800 to 1,000 clips / 20 to 30 volunteers / 6 environments; augmentation reframed as regularisation only |
| 3 | F1 >= 0.8 on this split does not prove robustness | Removed all pre-committed F1 targets; added 5-fold CV, subject-disjoint splits, OOD evaluation, Stage 1 vs Stage 2 comparison table |
| 4 | Battery will be a serious issue; do not fix the solution before starting | Reframed battery as anticipated risk; removed "3 to 5% per hour" target; empirical Month 1 benchmark; added noise cancellation stage (FR-3.0); rewrote LM-9 and OB-5 |

Unchanged since v1.0: Vision Statement, Module 1, Modules 6 to 9, WBS/Gantt, Mockups, Appendices A and B.

---

## 19. Vision statement

For women in Pakistan and South Asia who face the routine possibility of harassment or assault in public spaces, who need a phone that can call for help on their behalf when they are unable to act, the SheAlert Android application is a fully passive AI-powered personal safety system that continuously monitors motion and audio, detects threatening situations automatically, and sends an emergency alert with GPS location, a short audio clip, and an optional camera snapshot to pre-registered contacts, without requiring any action from the user during a crisis. Unlike every other safety app in Pakistan today, which require a button press, shake, or key hold, SheAlert is built on the recognition that someone in real danger usually cannot do those things.

---

## 20. Novelty claims

1. Passive multi-modal sensing (motion + audio + optional camera) on a standard Android device, end to end, on-device.
2. First labelled Urdu assault-distress audio dataset (bachao, madad karo, choro mujhe), released under CC-BY-4.0. Backed by a search of IEEE Xplore, ACM DL, arXiv, and Google Scholar with query: ("Urdu") AND ("distress" OR "scream" OR "panic" OR "women safety") AND ("audio" OR "dataset" OR "classification"). Date of search to be reported in the thesis.
3. Original Struggle/Resistance and Forced Movement smartphone-IMU dataset (no public dataset found).
4. Formal ablation of modality fusion, noise cancellation, confirmation window, and autoencoder vs multiclass approach.
5. Placement variability study (pocket, bag, hand) for smartphone-based threat detection.

---

## 21. References (as cited in the proposal)

1. Anguita et al., "A public domain dataset for human activity recognition using smartphones," ESANN, 2013. (UCI HAR)
2. Atrey et al., "Multimodal fusion for multimedia analysis: a survey," Multimedia Systems, 2010.
3. Baltrusaitis, Ahuja, Morency, "Multimodal machine learning: a survey and taxonomy," IEEE TPAMI, 2019.
4. Bovin et al., "Tonic immobility mediates the influence of peritraumatic fear...," J. Traumatic Stress, 2014.
5. Cao et al., "CREMA-D," IEEE Trans. Affective Computing, 2014.
6. Chand and Paul, "A review on women safety mobile applications," IJCA, 2015.
7. Fawcett, "An introduction to ROC analysis," Pattern Recognition Letters, 2006.
8. Google, "YAMNet: an audio event classifier," TF Hub, 2019.
9. Hershey et al., "CNN architectures for large-scale audio classification," ICASSP, 2017.
10. Hochreiter and Schmidhuber, "Long short-term memory," Neural Computation, 1997.
11. Kumar and Lenin Fred, "A survey on Indian women safety mobile applications," IJPAM, 2018.
12. Kwapisz, Weiss, Moore, "Activity recognition using cell phone accelerometers," SIGKDD Explorations, 2011. (WISDM)
13. Lane et al., "DeepX: a software accelerator for low-power deep learning inference on mobile devices," IPSN, 2016.
14. Malhotra et al., "LSTM-based encoder-decoder for multi-sensor anomaly detection," ICML Workshop, 2016.
15. Moller, Sondergaard, Helstrom, "Tonic immobility during sexual assault...," Acta Obstet Gynecol Scand, 2017.
16. Pakistan Bureau of Statistics, "7th Population and Housing Census 2023."
17. Sandler et al., "MobileNetV2: inverted residuals and linear bottlenecks," CVPR, 2018.
18. Yarrabothu and Thota, "Abhaya: an Android app for the safety of women," INDICON, 2015.
19. Android Developers, "Foreground services: microphone type," 2024.
20. Akhtar, Jahangir, Ain, "UrduSER: a dataset for Urdu speech emotion recognition," Mendeley Data v3, DOI 10.17632/jcpfjnk5c2.3, Dec 2024.
21. Attia et al., "UrduSpeech: a large-scale multi-domain Urdu speech corpus," arXiv:2605.17846, 2026. https://huggingface.co/datasets/ASLP-lab/UrduSpeech
22. Zaheer, Zehra, Hussain, Ali, "SEMOUR: a scripted emotional speech corpus for Urdu," ACM Multimedia, 2021. https://acoustics-lab.itu.edu.pk
23. Chatzaki et al., "Human daily activity and fall recognition using a smartphone's acceleration sensor," ICT4AWE, 2017. (MobiAct)
24. Micucci, Mobilio, Napoletano, "UniMiB SHAR," Applied Sciences 7(10), 1101, 2017.

---

## 22. Glossary

| Term | Meaning |
|---|---|
| HAR | Human Activity Recognition |
| VAD | Voice Activity Detection |
| YAMNet | Google pretrained audio event classifier (MobileNet-based, AudioSet) |
| TFLite | TensorFlow Lite, on-device inference runtime |
| Late fusion | Combine per-modality scores after each model has produced its own output |
| ROC | Receiver Operating Characteristic curve, used to pick the fusion threshold |
| OOD | Out-of-distribution (held-out acoustic environment) |
| Tonic immobility | Involuntary freezing during extreme fear, a reason manual triggers fail |
| Dead man's switch PIN | Cancel requires a PIN so an attacker cannot simply tap Cancel |
| FCM | Firebase Cloud Messaging |
| OEM whitelist | Prompts to exempt the app from aggressive vendor battery killers (MIUI, One UI, ColorOS, EMUI) |
| Geofence | User-defined home radius; audio pipeline pauses inside it |
