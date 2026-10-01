# SheAlert Documentation Portal

Welcome to the central documentation hub for **SheAlert: An Automatic Women Safety System Using Sensors, Audio, and Camera**.

This documentation is maintained side-by-side with the codebase so that all team members, academic supervisors, and evaluation panels can understand the system architecture, mathematical formulations, engineering decisions, and current progress without having to decipher thousands of lines of raw code.

---

## 📚 Document Index

| Document | Purpose | Target Audience |
|---|---|---|
| [System Architecture Overview](file:///f:/SheAlert-Women-Safety-System/docs/ARCHITECTURE_OVERVIEW.md) | High-level system design, multi-modal pipeline, functional requirements (FR), and module boundaries. | All Team Members, Supervisors, Panel |
| [Module 3 & 4 Audio Pipeline Specification](file:///f:/SheAlert-Women-Safety-System/docs/MODULE_3_AUDIO_PIPELINE.md) | In-depth engineering documentation of the audio consolidation, quality control (QC), clipping rescue, speaker grouping, and 5-fold stratified cross-validation. | Moaz (Author), Team, ML Reviewers |
| [FYP Panel Defense & Presentation Cheatsheet](file:///f:/SheAlert-Women-Safety-System/docs/PANEL_PRESENTATION_CHEATSHEET.md) | Quick-reference facts, verified metrics, architectural rationales, and bullet-proof answers to anticipated panel questions. | Defense Panel, Team Presenters |
| [Collaboration & Side-by-Side Workflow](file:///f:/SheAlert-Women-Safety-System/docs/CONTRIBUTING_AND_WORKFLOW.md) | Git conventions, branch management, coding standards, and the protocol for keeping docs synchronized with code. | Team (Moaz, Mahad, Hunain) |
| [Engineering Work Log & Changelog](file:///f:/SheAlert-Women-Safety-System/docs/WORK_LOG.md) | Chronological log of development milestones, algorithmic decisions, and completed tasks. | Team, Supervisors |
| [Proposal Context & Requirements Spec](file:///f:/SheAlert-Women-Safety-System/SHEALERT_CONTEXT.md) | Verbatim functional specification and requirement IDs (FR-1.0 to LM-5) from the FYP proposal v2.1. | Reference |

---

## 👥 Team & Module Responsibilities

SheAlert is developed by BS Artificial Intelligence students at Air University, Islamabad (Department of Creative Technologies):

* **Mahad Jokhio (231242)**: Motion & Sensor AI (*Module 1: Sensor Collection & Normalization, Module 2: Multiclass LSTM*).
* **Moaz Hassan Khan Manj (231168)**: Audio Pipeline & Fusion Engine (*Module 3: Pre-processing & Cascaded VAD, Module 4: Urdu Distress Audio Fine-tuning, Module 5: Context-Adaptive Late Fusion*).
* **Hunain Ahmed (231156)**: Confirmation UI, SOS Pipeline & App (*Module 6: Confirmation Window, Module 7: Camera Fail-Safe, Module 8: Alert Dispatch, Module 9: Flutter App & Validation*).

**Supervisory Team:**
* **Dr. Samana Batool** (Supervisor, Assistant Professor)
* **Dr. Madiha Yousaf** (Co-Supervisor, Assistant Professor)
* **Ms. Maryam Aamir Ashraf** (Industrial Supervisor, Associate Data Scientist)

---

## 🚀 Quick Verification

To verify that the current audio pipeline passes all data integrity, group leakage, and QC tests:

```bash
cd shealert_audio_pipeline
pytest -v
```

All 28 automated test suites should pass cleanly.
