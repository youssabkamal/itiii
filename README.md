# RAG-Based E-Commerce Customer Support Chatbot

An end-to-end multi-stage NLP system that provides grounded, accurate, and tone-appropriate customer support responses for retail and e-commerce platforms.

---

## System Architecture

The chatbot processes each incoming message through four sequential stages before generating a final response:

```
                  +-------------------------+
                  |  Incoming Customer Query|
                  +------------+------------+
                               |
                               v
                  +-------------------------+
                  | Stage 1: Language ID    | -> Identifies language (20 languages)
                  +------------+------------+
                               |
                               v
                  +-------------------------+
                  | Stage 2: Sentiment/Tone | -> Negative (Frustrated), Neutral, Positive
                  +------------+------------+
                               |
                               v
                  +-------------------------+
                  | Stage 3: Intent Routing | -> 7 Condensed Routing Categories
                  +------------+------------+
                               |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
  [greeting / small talk]  [out_of_scope]      [retail support / complaint]
        |                      |                      |
  Direct Response        Domain Scope Guard           v
  (No Retrieval)         (No Retrieval)         +-------------------------+
                                                | Priority Routing Check  |
                                                | (Escalation / Apology)  |
                                                +------------+------------+
                                                             |
                                                             v
                                                +-------------------------+
                                                | Stage 4: Q&A RAG        |
                                                | ChromaDB + all-MiniLM   |
                                                +------------+------------+
                                                             |
                                                             v
                                                +-------------------------+
                                                | Tone-Adjusted LLM /     |
                                                | Grounded Generation     |
                                                +-------------------------+
```

---

## Project Modules

### 1. Language Detection
- **Dataset:** `papluca/language-identification` (20 languages, 90k samples).
- **Architecture:** Sublinear TF-IDF Vectorizer with character n-grams (`analyzer='char_wb', ngram_range=(2, 5)`).
- **Classifier:** Calibrated Logistic Regression (`C=5.0`).
- **Enhancements:** Text normalization (NFKC Unicode, lowercase), subword character boundaries, and confidence thresholding for short input messages.
- **Accuracy:** 99.58% test accuracy.

### 2. Sentiment / Emotion Classifier
- **Dataset:** `dair-ai/emotion` (20k samples across 6 emotions).
- **Label Mapping:** 
  - **Negative:** Sadness (0), Anger (3), Fear (4).
  - **Neutral:** Surprise (5) and informational support queries.
  - **Positive:** Joy (1), Love (2).
- **Architecture:** PyTorch Bidirectional LSTM (`BiLSTMSentimentNet`) with Embedding layer, dropout, and Linear classification head.
- **Accuracy:** 95.20% test accuracy.

### 3. Intent Classifier
- **Dataset:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (26,872 gold-labeled pairs).
- **Consolidated Categories:**
  1. `greeting`: Small talk, greetings, gratitude, and goodbyes (bypasses retrieval).
  2. `order_status`: `track_order`, `delivery_options`, `delivery_period`.
  3. `order_management`: `cancel_order`, `change_order`, `place_order`, `change_shipping_address`, `set_up_shipping_address`, `check_cancellation_fee`.
  4. `billing_and_refunds`: `check_invoice`, `get_invoice`, `get_refund`, `track_refund`, `check_refund_policy`, `payment_issue`, `check_payment_methods`.
  5. `account_management`: `create_account`, `edit_account`, `delete_account`, `switch_account`, `recover_password`, `registration_problems`, `newsletter_subscription`.
  6. `complaint`: `complaint`, `review`, `contact_customer_service`, `contact_human_agent` (triggers priority escalation).
  7. `out_of_scope`: General queries outside retail domain.
- **Classifier:** Multi-class Logistic Regression with word n-grams (1, 3).

### 4. Q&A RAG Pipeline
- **Vector Database:** Local persistent ChromaDB (`models/chroma_db`).
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Knowledge Base:** Bitext instruction/response pairs embedded for semantic search.
- **Prompt Structure:**
  ```
  System: "You are a helpful, professional customer support assistant
    for an online retailer. Answer the customer's question using ONLY
    the information in the retrieved support responses below. If the
    customer sounds frustrated ({detected_sentiment}), acknowledge
    that before answering. If the retrieved context does not cover
    the question, say so honestly and offer to escalate to a human
    agent rather than guessing."
  Context (retrieved past support responses):
  {retrieved_chunk_1}
  {retrieved_chunk_2}
  {retrieved_chunk_3}
  Customer question: "{user_message}"
  ```
- **Generation:** Groq API (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) with automatic offline fallback to grounded context synthesizer.

---

## Routing and Escalation Logic

- **Small Talk (`greeting`):** Answered immediately with helpful guidance; avoids unnecessary vector retrieval.
- **Out of Scope (`out_of_scope`):** Gently guides user back to supported e-commerce topics.
- **Frustrated Sentiment (`negative`):** Prepend empathetic acknowledgment and apology.
- **Complaints & Grievances:** Assigned a unique ticket ID (`ESC-XXXXXXXX`), flagged for human supervisor escalation, and provided with grounded policy resolution.
- **Multi-Lingual Inquiries:** Automatically identifies language and provides assistance with language tags.

---

## Repository Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── train_all.py
├── pipeline.py
├── app.py
├── language_detection.ipynb
├── sentiment_classifier.ipynb
├── intent_classifier.ipynb
├── rag_pipeline.ipynb
├── notebooks/
│   ├── language_detection.ipynb
│   ├── sentiment_classifier.ipynb
│   ├── intent_classifier.ipynb
│   └── rag_pipeline.ipynb
├── models/
│   ├── language_detector.pkl
│   ├── sentiment_classifier.pt
│   ├── sentiment_meta.pkl
│   ├── intent_classifier.pkl
│   └── chroma_db/
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── app.js
```

---

## Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Train Models and Build Vector DB
```bash
python train_all.py
```

### 3. Launch Web Application
```bash
python app.py
```
Open your browser at `http://127.0.0.1:8000` to interact with the chatbot interface.
