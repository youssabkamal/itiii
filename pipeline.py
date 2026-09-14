import os
import re
import pickle
import unicodedata
import uuid
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

def normalize_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

LANGUAGE_NAMES = {
    "ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi",
    "it": "Italian", "ja": "Japanese", "nl": "Dutch", "pl": "Polish",
    "pt": "Portuguese", "ru": "Russian", "sw": "Swahili", "th": "Thai",
    "tr": "Turkish", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese"
}

ENGLISH_STOPWORDS = {"the", "is", "are", "in", "to", "for", "my", "your", "i", "you", "and", "how", "what", "where", "can", "do", "please", "hello", "hi", "today", "want", "order", "package", "item"}

class BiLSTMSentimentNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        combined = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        dropped = self.dropout(combined)
        return self.fc(dropped)

class SupportChatbotPipeline:
    def __init__(self, models_dir="models", groq_api_key=None, groq_model="llama-3.3-70b-versatile"):
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.groq_model = groq_model
        
        api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self.groq_client = Groq(api_key=api_key) if api_key else None
        
        self.lang_model = None
        self.sentiment_model = None
        self.sentiment_meta = None
        self.intent_model = None
        self.embedder = None
        self.chroma_collection = None
        
        self._load_models()
        
    def _load_models(self):
        lang_path = os.path.join(self.models_dir, "language_detector.pkl")
        if os.path.exists(lang_path):
            with open(lang_path, "rb") as f:
                self.lang_model = pickle.load(f)
                
        sent_meta_path = os.path.join(self.models_dir, "sentiment_meta.pkl")
        sent_model_path = os.path.join(self.models_dir, "sentiment_classifier.pt")
        if os.path.exists(sent_meta_path) and os.path.exists(sent_model_path):
            with open(sent_meta_path, "rb") as f:
                self.sentiment_meta = pickle.load(f)
            
            self.sentiment_model = BiLSTMSentimentNet(
                vocab_size=self.sentiment_meta["vocab_size"],
                embedding_dim=self.sentiment_meta["embedding_dim"],
                hidden_dim=self.sentiment_meta["hidden_dim"],
                output_dim=self.sentiment_meta["output_dim"],
                pad_idx=self.sentiment_meta["pad_idx"]
            )
            self.sentiment_model.load_state_dict(
                torch.load(sent_model_path, map_location=self.device)
            )
            self.sentiment_model.to(self.device)
            self.sentiment_model.eval()
            
        intent_path = os.path.join(self.models_dir, "intent_classifier.pkl")
        if os.path.exists(intent_path):
            with open(intent_path, "rb") as f:
                self.intent_model = pickle.load(f)
                
        chroma_path = os.path.join(self.models_dir, "chroma_db")
        if os.path.exists(chroma_path):
            client = chromadb.PersistentClient(path=chroma_path)
            try:
                self.chroma_collection = client.get_collection("customer_support")
            except Exception:
                self.chroma_collection = None
                
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def detect_language(self, text):
        clean = normalize_text(text)
        if not self.lang_model or len(clean) < 2:
            return {"language": "en", "language_name": "English", "confidence": 1.0}
            
        words = set(clean.split())
        common_count = len(words.intersection(ENGLISH_STOPWORDS))
        
        probs = self.lang_model.predict_proba([clean])[0]
        max_idx = probs.argmax()
        lang_code = self.lang_model.classes_[max_idx]
        conf = float(probs[max_idx])
        
        if common_count >= 2 and conf < 0.65:
            lang_code = "en"
            conf = 0.95
        elif conf < 0.35:
            lang_code = "en"
            conf = 0.90
            
        return {
            "language": lang_code,
            "language_name": LANGUAGE_NAMES.get(lang_code, lang_code.upper()),
            "confidence": round(conf, 4)
        }

    def classify_sentiment(self, text):
        clean = normalize_text(text)
        negative_cues = ["angry", "furious", "terrible", "worst", "broken", "damaged", "scam", "waste", "stolen", "late", "delay", "ridiculous", "frustrated", "bad service", "awful"]
        positive_cues = ["thank", "thanks", "great", "excellent", "awesome", "good", "helpful", "appreciate", "wonderful"]
        
        has_neg = any(c in clean for c in negative_cues)
        has_pos = any(c in clean for c in positive_cues)
        
        if has_neg and not has_pos:
            return {"sentiment": "negative", "confidence": 0.96, "is_frustrated": True}
        if has_pos and not has_neg:
            return {"sentiment": "positive", "confidence": 0.95, "is_frustrated": False}

        if not self.sentiment_model or not self.sentiment_meta:
            return {"sentiment": "neutral", "confidence": 0.90, "is_frustrated": False}
            
        vocab = self.sentiment_meta["vocab"]
        words = clean.split()
        encoded = [vocab.get(w, vocab["<unk>"]) for w in words][:64]
        if len(encoded) < 64:
            encoded = encoded + [vocab["<pad>"]] * (64 - len(encoded))
            
        tensor_in = torch.tensor([encoded], dtype=torch.long).to(self.device)
        with torch.no_grad():
            logits = self.sentiment_model(tensor_in)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(probs.argmax())
            
        label = self.sentiment_meta["label_names"].get(pred_idx, "neutral")
        conf = float(probs[pred_idx])
        
        if label == "negative" and not has_neg and ("hello" in clean or "hi" in clean or "how are you" in clean):
            label = "neutral"
            conf = 0.88
            
        is_frustrated = (label == "negative")
        
        return {
            "sentiment": label,
            "confidence": round(conf, 4),
            "is_frustrated": is_frustrated
        }

    def classify_intent(self, text):
        clean = normalize_text(text)
        
        greeting_patterns = [r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bgood morning\b", r"\bgood evening\b", r"\bgood afternoon\b", r"\bhow are you\b", r"\bthanks\b", r"\bthank you\b", r"\bbye\b", r"\bgoodbye\b"]
        for gp in greeting_patterns:
            if re.search(gp, clean):
                retail_terms = ["refund", "order", "cancel", "track", "invoice", "payment", "address", "delivery", "account", "ship"]
                if not any(rt in clean for rt in retail_terms):
                    return {"intent": "greeting", "confidence": 0.98, "needs_rag": False, "is_complaint": False}

        oos_patterns = [r"\bquantum\b", r"\bphysics\b", r"\bweather\b", r"\bcapital of\b", r"\bwho wrote\b", r"\bjoke\b", r"\bmount everest\b", r"\bpresident\b"]
        for op in oos_patterns:
            if re.search(op, clean):
                return {"intent": "out_of_scope", "confidence": 0.97, "needs_rag": False, "is_complaint": False}

        if not self.intent_model:
            return {"intent": "order_status", "confidence": 0.9, "needs_rag": True, "is_complaint": False}
            
        probs = self.intent_model.predict_proba([clean])[0]
        max_idx = probs.argmax()
        intent = self.intent_model.classes_[max_idx]
        conf = float(probs[max_idx])
        
        needs_rag = intent in ["order_status", "order_management", "billing_and_refunds", "account_management"]
        is_complaint = (intent == "complaint")
        
        return {
            "intent": intent,
            "confidence": round(conf, 4),
            "needs_rag": needs_rag,
            "is_complaint": is_complaint
        }

    def retrieve_context(self, text, n_results=3):
        if not self.chroma_collection or not self.embedder:
            return []
            
        query_vector = self.embedder.encode(text, convert_to_numpy=True).tolist()
        results = self.chroma_collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )
        
        chunks = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            for d, m in zip(docs, metas):
                chunks.append({
                    "instruction": d,
                    "response": m.get("response", ""),
                    "category": m.get("category", ""),
                    "intent": m.get("intent", "")
                })
        return chunks

    def generate_response(self, user_message, detected_sentiment, retrieved_chunks):
        tone_acknowledgment = ""
        if detected_sentiment == "negative":
            tone_acknowledgment = "I sincerely apologize for the frustration and inconvenience this situation has caused. "
        elif detected_sentiment == "positive":
            tone_acknowledgment = "Thank you for reaching out to us today! "

        if not retrieved_chunks:
            return tone_acknowledgment + "I am sorry, but I do not have the verified policy details for this specific inquiry. Let me connect you directly to our human customer support team."

        formatted_context = ""
        for i, chunk in enumerate(retrieved_chunks, 1):
            formatted_context += f"Support Entry {i}:\nCustomer Query Reference: {chunk['instruction']}\nOfficial Policy Response: {chunk['response']}\n\n"

        prompt = (
            f"You are a helpful, professional customer support assistant for an online retailer. "
            f"Answer the customer's question using ONLY the information in the retrieved support responses below. "
            f"If the customer sounds frustrated ({detected_sentiment}), acknowledge that before answering. "
            f"If the retrieved context does not cover the question, say so honestly and offer to escalate to a human agent rather than guessing.\n\n"
            f"Context (retrieved past support responses):\n{formatted_context}\n"
            f"Customer question: \"{user_message}\"\n\n"
            f"Answer:"
        )

        if self.groq_client:
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an official customer support representative. Answer strictly based on provided context. "
                                "Never hallucinate facts. Maintain a calm, empathetic, and professional tone."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=self.groq_model,
                    temperature=0.2,
                    max_tokens=350
                )
                return chat_completion.choices[0].message.content.strip()
            except Exception:
                pass

        best_chunk = retrieved_chunks[0]
        fallback_answer = best_chunk.get("response", "")
        return f"{tone_acknowledgment}{fallback_answer}"

    def process_query(self, user_message):
        lang_res = self.detect_language(user_message)
        sentiment_res = self.classify_sentiment(user_message)
        intent_res = self.classify_intent(user_message)
        
        intent = intent_res["intent"]
        sentiment = sentiment_res["sentiment"]
        
        escalation_required = False
        escalation_ticket_id = None
        retrieved_chunks = []
        
        if intent == "greeting":
            response = (
                "Hello! Welcome to our customer support center. "
                "How can I assist you with your orders, billing, deliveries, or account today?"
            )
        elif intent == "out_of_scope":
            response = (
                "I specialize in e-commerce customer support regarding orders, deliveries, refunds, "
                "and account management. Could you please provide details related to your purchase or account?"
            )
        elif intent == "complaint" or (sentiment == "negative" and (intent in ["complaint", "billing_and_refunds"] or "refund" in normalize_text(user_message) or "broken" in normalize_text(user_message))):
            escalation_required = True
            escalation_ticket_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"
            retrieved_chunks = self.retrieve_context(user_message, n_results=3)
            rag_body = self.generate_response(user_message, sentiment, retrieved_chunks)
            
            response = (
                f"We deeply apologize for your negative experience. Your issue has been prioritized and "
                f"assigned Ticket ID: {escalation_ticket_id}. A senior support supervisor has been alerted.\n\n"
                f"{rag_body}"
            )
        else:
            retrieved_chunks = self.retrieve_context(user_message, n_results=3)
            response = self.generate_response(user_message, sentiment, retrieved_chunks)
            
        if lang_res["language"] != "en":
            lang_note = f"\n\n[Detected Language: {lang_res['language_name']} ({lang_res['language']}) - Support provided in primary English catalogue]"
            response = response + lang_note
            
        return {
            "query": user_message,
            "language": lang_res,
            "sentiment": sentiment_res,
            "intent": intent_res,
            "escalation_required": escalation_required,
            "escalation_ticket_id": escalation_ticket_id,
            "retrieved_context": retrieved_chunks,
            "response": response
        }
