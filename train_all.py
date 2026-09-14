import os
import re
import pickle
import unicodedata
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import chromadb
from sentence_transformers import SentenceTransformer

os.makedirs("models", exist_ok=True)

def normalize_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def train_language_detector():
    if os.path.exists("models/language_detector.pkl"):
        print("Language Detector already trained.")
        return
        
    raw_train = load_dataset("papluca/language-identification", split="train")
    raw_test = load_dataset("papluca/language-identification", split="test")
    
    train_df = pd.DataFrame(raw_train)
    test_df = pd.DataFrame(raw_test)
    
    train_df["text"] = train_df["text"].apply(normalize_text)
    test_df["text"] = test_df["text"].apply(normalize_text)
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=50000,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            C=5.0,
            max_iter=1000,
            solver="lbfgs"
        ))
    ])
    
    pipeline.fit(train_df["text"], train_df["labels"])
    
    preds = pipeline.predict(test_df["text"])
    acc = accuracy_score(test_df["labels"], preds)
    print(f"Language Detector Test Accuracy: {acc:.4f}")
    
    with open("models/language_detector.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    
    return pipeline

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

class TextDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx, max_len=64):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        words = self.texts[idx].split()
        encoded = [self.word_to_idx.get(w, self.word_to_idx["<unk>"]) for w in words][:self.max_len]
        if len(encoded) < self.max_len:
            encoded = encoded + [self.word_to_idx["<pad>"]] * (self.max_len - len(encoded))
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def train_sentiment_classifier():
    if os.path.exists("models/sentiment_classifier.pt") and os.path.exists("models/sentiment_meta.pkl"):
        print("Sentiment Classifier already trained.")
        return
        
    raw_train = load_dataset("dair-ai/emotion", split="train")
    raw_test = load_dataset("dair-ai/emotion", split="test")
    
    emotion_map = {
        0: 0,
        1: 2,
        2: 2,
        3: 0,
        4: 0,
        5: 1
    }
    
    extra_neutral = [
        "Where is my package located right now?",
        "Can I check my current invoice balance?",
        "What are the shipping options for my address?",
        "How do I reset my account password?",
        "Is there an option to change the delivery date?",
        "Please provide details about your return policy.",
        "I need to download a copy of my receipt.",
        "Could you clarify the cancellation fee policy?"
    ]
    
    train_texts = [normalize_text(t) for t in raw_train["text"]]
    train_labels = [emotion_map[label] for label in raw_train["label"]]
    
    for en in extra_neutral:
        train_texts.append(normalize_text(en))
        train_labels.append(1)
        
    test_texts = [normalize_text(t) for t in raw_test["text"]]
    test_labels = [emotion_map[label] for label in raw_test["label"]]
    
    vocab = {"<pad>": 0, "<unk>": 1}
    for text in train_texts:
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(vocab)
                
    train_dataset = TextDataset(train_texts, train_labels, vocab)
    test_dataset = TextDataset(test_texts, test_labels, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMSentimentNet(
        vocab_size=len(vocab),
        embedding_dim=64,
        hidden_dim=64,
        output_dim=3,
        pad_idx=vocab["<pad>"]
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    
    epochs = 4
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
    test_acc = correct / total
    print(f"Sentiment Classifier Test Accuracy: {test_acc:.4f}")
    
    torch.save(model.state_dict(), "models/sentiment_classifier.pt")
    
    meta = {
        "vocab": vocab,
        "label_names": {0: "negative", 1: "neutral", 2: "positive"},
        "vocab_size": len(vocab),
        "embedding_dim": 64,
        "hidden_dim": 64,
        "output_dim": 3,
        "pad_idx": vocab["<pad>"]
    }
    with open("models/sentiment_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

def map_intent_category(fine_intent):
    intent_mapping = {
        "track_order": "order_status",
        "delivery_options": "order_status",
        "delivery_period": "order_status",
        
        "cancel_order": "order_management",
        "change_order": "order_management",
        "place_order": "order_management",
        "change_shipping_address": "order_management",
        "set_up_shipping_address": "order_management",
        "check_cancellation_fee": "order_management",
        
        "check_invoice": "billing_and_refunds",
        "get_invoice": "billing_and_refunds",
        "get_refund": "billing_and_refunds",
        "track_refund": "billing_and_refunds",
        "check_refund_policy": "billing_and_refunds",
        "payment_issue": "billing_and_refunds",
        "check_payment_methods": "billing_and_refunds",
        
        "create_account": "account_management",
        "edit_account": "account_management",
        "delete_account": "account_management",
        "switch_account": "account_management",
        "recover_password": "account_management",
        "registration_problems": "account_management",
        "newsletter_subscription": "account_management",
        
        "complaint": "complaint",
        "review": "complaint",
        "contact_customer_service": "complaint",
        "contact_human_agent": "complaint"
    }
    return intent_mapping.get(fine_intent, "out_of_scope")

def train_intent_classifier():
    if os.path.exists("models/intent_classifier.pkl"):
        print("Intent Classifier already trained.")
        return
        
    raw_dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    df = pd.DataFrame(raw_dataset)
    
    df["clean_text"] = df["instruction"].apply(normalize_text)
    df["mapped_intent"] = df["intent"].apply(map_intent_category)
    
    greetings = [
        "hello", "hi there", "hey", "good morning", "good afternoon",
        "good evening", "hi bot", "hey support", "thank you so much",
        "thanks for the help", "appreciate it", "goodbye", "bye", "see you later"
    ]
    out_of_scope = [
        "what is the capital of italy", "tell me a funny joke",
        "who won the world cup in 1998", "how to write python code",
        "what is the weather like today", "can you write an essay about space",
        "how tall is mount everest", "who is the president of france"
    ]
    
    extra_rows = []
    for g in greetings:
        extra_rows.append({"clean_text": normalize_text(g), "mapped_intent": "greeting"})
    for o in out_of_scope:
        extra_rows.append({"clean_text": normalize_text(o), "mapped_intent": "out_of_scope"})
        
    df_extra = pd.DataFrame(extra_rows)
    df_combined = pd.concat([df[["clean_text", "mapped_intent"]], df_extra], ignore_index=True)
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=25000,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            C=3.0,
            max_iter=1000,
            solver="lbfgs"
        ))
    ])
    
    pipeline.fit(df_combined["clean_text"], df_combined["mapped_intent"])
    
    with open("models/intent_classifier.pkl", "wb") as f:
        pickle.dump(pipeline, f)
        
    print("Intent Classifier trained and saved.")
    return pipeline

def build_rag_vector_db():
    raw_dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    df = pd.DataFrame(raw_dataset)
    
    df_dedup = df.drop_duplicates(subset=["instruction"]).copy()
    
    sample_dfs = []
    for intent_name, group in df_dedup.groupby("intent"):
        sample_dfs.append(group.head(100))
        
    sample_df = pd.concat(sample_dfs, ignore_index=True)
    print(f"Building Chroma vector store with {len(sample_df)} representative support entries...")
    
    client = chromadb.PersistentClient(path="models/chroma_db")
    
    try:
        client.delete_collection(name="customer_support")
    except Exception:
        pass
        
    collection = client.create_collection(
        name="customer_support",
        metadata={"hnsw:space": "cosine"}
    )
    
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    instructions = sample_df["instruction"].tolist()
    responses = sample_df["response"].tolist()
    intents = sample_df["intent"].tolist()
    categories = sample_df["category"].tolist()
    
    batch_size = 256
    for i in range(0, len(instructions), batch_size):
        b_inst = instructions[i:i + batch_size]
        b_resp = responses[i:i + batch_size]
        b_int = intents[i:i + batch_size]
        b_cat = categories[i:i + batch_size]
        
        b_ids = [f"doc_{j}" for j in range(i, i + len(b_inst))]
        b_meta = [{"response": r, "intent": it, "category": c} for r, it, c in zip(b_resp, b_int, b_cat)]
        
        b_embeddings = embedder.encode(b_inst, convert_to_numpy=True).tolist()
        
        collection.add(
            documents=b_inst,
            embeddings=b_embeddings,
            metadatas=b_meta,
            ids=b_ids
        )
        
    print(f"ChromaDB vector collection successfully built with {collection.count()} entries.")

if __name__ == "__main__":
    print("Stage 1: Checking Language Detector...")
    train_language_detector()
    
    print("Stage 2: Checking Sentiment Classifier...")
    train_sentiment_classifier()
    
    print("Stage 3: Checking Intent Classifier...")
    train_intent_classifier()
    
    print("Stage 4: Building RAG Vector Store...")
    build_rag_vector_db()
    
    print("All models and vector database ready.")
