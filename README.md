# PasswordTensor  
### Analyzing and Explaining Password Strength Using Tensor Decomposition

## Project Overview
**PasswordTensor** is a cybersecurity-focused project that analyzes and explains password strength using **syntactic and semantic features** combined with **tensor decomposition (PARAFAC2)**.  
Unlike traditional rule-based password checkers, this system provides **data-driven, explainable strength classification**.

The project is implemented using **Python** and includes a **Flask-based web interface** for real-time password strength evaluation.

---

## Key Objectives
- Analyze password strength beyond basic rules
- Extract syntactic and semantic password features
- Apply **PARAFAC2 tensor decomposition** to identify latent strength patterns
- Classify passwords into **Weak, Medium, Strong**
- Provide **confidence score, explanations, and warning flags**
- Enable real-time evaluation via a web interface

---

## Core Concepts Used
- Syntactic feature extraction (length, digits, transitions, character classes)
- Semantic analysis using NLP (noun, verb, adjective detection)
- Min–Max feature normalization
- Tensor construction (Password × Feature × Strength)
- PARAFAC2 tensor decomposition
- Cosine similarity–based distance analysis
- Rule-based overrides for leaked and structurally weak passwords

---

## Project Structure
    ├── app.py                     # Flask web application entry point
    ├── password_analyser.py       # Core password strength prediction logic
    ├── py2.py                     # Feature extraction & tensor analysis (experiment)
    ├── py3.py                     # Alternative / optimized tensor implementation
    ├── templates/
    │   └── index.html             # Frontend UI for password input
    ├── datasets/
    │   ├── 10000_common_passwords.csv
    │   └── rockyou.csv
    ├── FINAL__MINI_PROJECT_SUBMISSION_REPORT.pdf
    ├── README.md

---

## Technologies & Libraries
- **Python 3**
- **Flask**
- **Pandas, NumPy**
- **NLTK**
- **Scikit-learn**
- **TensorLy**
- **Matplotlib**

---

## How the System Works
1. User inputs a password
2. Syntactic & semantic features are extracted
3. Features are normalized
4. Password is projected into tensor factor space
5. Distance from strength cluster centers is computed
6. Strength label and confidence score are generated
7. Weakness flags and explanations are returned

---

## Sample Output
**Input Password:** `P@ssw0rd2024!`

**Output:**
- Strength: **Medium**
- Confidence: **72%**
- Flags:
  - Contains dictionary words
- Explanation:
  - Length
  - Character class diversity
  - Semantic word presence

---

## Dataset Information
- **10000 Common Passwords Dataset**
- **RockYou leaked password dataset**

> These datasets are used **strictly for academic and research purposes**  
> No offensive security, cracking, or misuse is intended.

---

## How to Run the Project

### Install Dependencies
    ```bash
    pip install flask pandas numpy nltk scikit-learn tensorly matplotlib
### Run the Web Application
    ```bash
    python app.py
### Open in Browser
    ```bash
    http://127.0.0.1:5000/



