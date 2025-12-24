# Physician Notetaker – Medical NLP Pipeline

This project implements an **end-to-end NLP pipeline** to process a **physician–patient conversation** and extract structured medical information, patient sentiment, and clinical documentation.

The solution is designed to be **practical, interpretable, and clinically safe**, prioritizing deterministic logic over speculative AI behavior.

---
## 🔧 Setup & Execution Instructions

This project is implemented using **Google Colab**.

### Option 1: Run in Google Colab (Recommended)
1. Open the notebook in Google Colab
2. Run all cells from top to bottom
3. Required libraries are installed automatically in the notebook

### Option 2: Run Locally
1. Ensure Python 3.9+ is installed
2. Install dependencies:
   ```bash
   pip install spacy transformers torch scikit-learn
   python -m spacy download en_core_web_sm

##  Input

A transcribed **doctor–patient conversation** describing a motor vehicle accident, subsequent symptoms, treatment, recovery progress, and physician assessment.

---

## 🧠 System Overview

All tasks operate on the **same medical transcript**, demonstrating how a single conversation can be transformed into multiple structured outputs.

```
Medical Conversation
        ↓
Task 1: Medical NLP Summarization
        ↓
Task 2: Sentiment & Intent Analysis
        ↓
Task 3: SOAP Note Generation (Bonus)
```

---

## 1. Medical NLP Summarization

### Task

Implement an NLP pipeline to **extract medical details** from the transcript.

### Deliverables & Implementation

#### 1. Named Entity Recognition (NER)

* Extracted medical entities:

  * **Symptoms** (e.g., neck pain, back pain)
  * **Diagnosis** (whiplash injury)
  * **Treatment** (painkillers, physiotherapy)
  * **Prognosis** (full recovery expected)
* Implemented using **spaCy-based NLP processing** and **rule-based logic** for reliability.

#### 2. Text Summarization

* Converted the transcript into a **structured medical summary (JSON)**.
* Focused on factual extraction rather than free-text generation to avoid hallucination.

#### 3. Keyword Extraction

* Identified key medical phrases such as:

  * *whiplash injury*
  * *physiotherapy sessions*
  * *occasional back pain*

### Handling Ambiguous or Missing Data

* The system **does not infer or guess** missing information.
* If a medical detail is not explicitly stated, it is marked as:

  * `"Not mentioned"` or `"Not explicitly mentioned"`

This mirrors real-world healthcare documentation standards.

### Example Output

```json
{
  "Patient_Name": "Janet Jones",
  "Symptoms": ["Neck pain", "Back pain"],
  "Diagnosis": "Whiplash injury",
  "Treatment": ["Physiotherapy", "Painkillers"],
  "Current_Status": "Occasional back pain",
  "Prognosis": "Full recovery expected within six months"
}
```

---

##  2. Sentiment & Intent Analysis

### Task

Detect **patient sentiment** and **intent** from the conversation.

### Deliverables & Implementation

#### 1. Sentiment Classification

* Implemented using a **pre-trained Transformer model**:

  * `distilbert-base-uncased-finetuned-sst-2-english`
* Sentiment is mapped to:

  * `Anxious`
  * `Neutral`
  * `Reassured`

**Important design choice:**
Sentiment analysis is applied **only to patient utterances**, not the full conversation, to:

* Avoid transformer token-length limits
* Ensure correct emotional interpretation

#### 2. Intent Detection

* Implemented using **rule-based logic**
* Identifies intents such as:

  * Reporting symptoms
  * Seeking reassurance
  * General health update

### Example Output

```json
{
  "Sentiment": "Reassured",
  "Intent": "Reporting symptoms"
}
```

### Fine-tuning Consideration (Theory)

* In production, BERT could be fine-tuned on labeled medical dialogue data.
* This implementation uses a pre-trained model due to assignment scope.

---

## 3. SOAP Note Generation 

### Task

Convert the transcript into a structured **SOAP note** format.

### Deliverables & Implementation

#### 1. Automated SOAP Note Generation

* Implemented using an **NLP-assisted, rule-based approach**
* Automatically maps transcript content into:

  * Subjective
  * Objective
  * Assessment
  * Plan

#### 2. Logical Mapping

* **Subjective:** Patient-reported symptoms and history
* **Objective:** Physician examination findings
* **Assessment:** Diagnosis and clinical evaluation
* **Plan:** Treatment and follow-up recommendations

#### 3. Clinical Readability

* Output is structured, readable, and suitable for clinical documentation.
* Missing information is explicitly marked rather than inferred.

### Example Output

```json
{
  "Subjective": {
    "Chief_Complaint": "Neck and back pain",
    "History_of_Present_Illness": "Pain following a car accident, initially severe and improved over time."
  },
  "Objective": {
    "Physical_Exam": "Full range of motion in cervical and lumbar spine, no tenderness.",
    "Observations": "Patient appears in normal health, normal gait."
  },
  "Assessment": {
    "Diagnosis": "Whiplash injury",
    "Severity": "Mild, improving"
  },
  "Plan": {
    "Treatment": "Continue physiotherapy as needed, use analgesics for pain relief.",
    "Follow-Up": "Patient to return if pain worsens or persists beyond six months."
  }
}
```

### Model Training Consideration (Theory)

* SOAP note generation could be trained using annotated transcripts labeled as S/O/A/P.
* A hybrid system combining **NER + sentence classification + rule-based safeguards** would improve accuracy in production.

---

## 🧪 Environment & Execution

* Developed and tested using **Google Colab**
* Notebook outputs are intentionally **not committed** to GitHub to keep the repository clean
* To reproduce results:

  * Open the notebook in Colab
  * Run all cells sequentially

---

## 🧰 Tech Stack

* Python
* spaCy
* HuggingFace Transformers

---

## 👤 Author

**Prathmesh Kulkarni**
GitHub: [https://github.com/pkkulk](https://github.com/pkkulk)

---
