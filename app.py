import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import csv
from datetime import datetime
from abc import ABC, abstractmethod
from transformers import pipeline

# ─────────────────────────────────────────────
#  OOP PILLARS
# ─────────────────────────────────────────────
class BaseReviewDetector(ABC):
    @abstractmethod
    def analyze(self, text: str): pass
    @abstractmethod
    def get_reasons(self, verdict, features, bert_prob, rule_score): pass
    def preprocess(self, text: str):
        return re.sub(r'\s+', ' ', text.strip())

class FeatureExtractor:
    FAKE_PHRASES = [
        "must purchase","one must","superior item","remarkable standard",
        "incredibly impressed","buy now","best best","amazing quality",
        "100% amazing","purchase immediately","highly recommend",
        "best product ever","do not hesitate","without a doubt",
        "exceeded my expectations","absolutely love","life changing",
        "perfect in every way","best purchase","totally worth",
        "don't miss","grab it now","order now","limited time",
        "act now","don't wait","best decision","zero regrets",
        "five stars","10/10","a must have","must have",
        "blown away","mind blowing","game changer","changed my life",
        "never been happier","could not be happier","top quality",
        "unbelievable quality","exceptional quality","outstanding quality",
    ]
    SPAM_WORDS = [
        "amazing","incredible","fantastic","wonderful","excellent",
        "superb","outstanding","brilliant","magnificent","phenomenal",
        "extraordinary","remarkable","spectacular","marvelous","terrific",
    ]
    ROMAN_URDU_FAKE_PHRASES = [
        "bilkul lelo","jaldi order karo","limited stock",
        "sab ko lena chahiye","bohat bohat acha","order karo abhi",
        "zabardast zabardast","best best best","itna acha itna acha",
        "kya cheez hai","kamaal hai","must have hai",
        "zindagi badal gayi","life changing experience",
        "zero regrets","totally worth","order abhi",
        "must purchase","one must purchase","immediately",
        "5 star","five stars","top quality top quality",
        "buy buy buy","kiya kiya kiya",
    ]
    ROMAN_URDU_SPAM_WORDS = [
        "zabardast","jabardast","kamaal","masha allah",
        "outstanding","shandar","lajawab","best","amazing",
        "incredible","phenomenal","spectacular","magnificent",
        "brilliant","superb","marvelous","terrific","extraordinary",
    ]

    def _is_roman_urdu(self, text: str) -> bool:
        roman_markers = [
            "hai","karo","lelo","acha","wala","theek","lekin",
            "mein","nahi","bhi","kya","toh","yaar","bhai",
            "bohat","zyada","bilkul","sab","yeh","meri","mere",
            "liya","tha","thi","kar","ho","se","ka","ki","ke",
        ]
        lower = text.lower()
        hits = sum(1 for m in roman_markers if re.search(r'\b' + m + r'\b', lower))
        return hits >= 3

    def extract(self, text: str) -> dict:
        words = text.split()
        word_count = len(words)
        excl_count = text.count('!')
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        unique_ratio = len(set([w.lower() for w in words])) / max(word_count, 1)
        repetition = int((pd.Series([w.lower() for w in words]).value_counts() > 2).sum())
        text_lower = text.lower()
        phrase_hits = sum(1 for p in self.FAKE_PHRASES if p in text_lower)
        spam_hits   = sum(1 for w in self.SPAM_WORDS if w in text_lower)
        is_roman = self._is_roman_urdu(text)
        ru_phrase_hits = sum(1 for p in self.ROMAN_URDU_FAKE_PHRASES if p in text_lower)
        ru_spam_hits   = sum(1 for w in self.ROMAN_URDU_SPAM_WORDS   if w in text_lower)
        superlative_count = len(re.findall(r'\b\w+est\b|\bmost \w+\b|\bvery \w+\b|\bso \w+\b', text_lower))
        all_caps_words    = sum(1 for w in words if w.isupper() and len(w) > 2)
        sentences         = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        avg_sentence_len  = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        has_url = 1 if re.search(r'http|www|\.com|\.net', text_lower) else 0
        return {
            "word_count": word_count, "excl_count": excl_count, "caps_ratio": caps_ratio,
            "avg_word_len": avg_word_len, "unique_ratio": unique_ratio, "repetition": repetition,
            "phrase_hits": phrase_hits, "spam_hits": spam_hits,
            "superlative_count": superlative_count, "all_caps_words": all_caps_words,
            "avg_sentence_len": avg_sentence_len, "has_url": has_url,
            "is_roman_urdu": int(is_roman), "ru_phrase_hits": ru_phrase_hits, "ru_spam_hits": ru_spam_hits,
        }

    def __rule_score(self, f: dict) -> float:
        score = 0.0
        if f["excl_count"] >= 4:       score += 40
        elif f["excl_count"] == 3:     score += 30
        elif f["excl_count"] == 2:     score += 20
        elif f["excl_count"] == 1:     score += 8
        if f["caps_ratio"] > 0.30:     score += 25
        elif f["caps_ratio"] > 0.20:   score += 15
        if f["all_caps_words"] >= 3:   score += 20
        elif f["all_caps_words"] >= 1: score += 10
        if f["unique_ratio"] < 0.40:   score += 25
        elif f["unique_ratio"] < 0.55: score += 12
        if f["repetition"] > 2:        score += 20
        elif f["repetition"] > 0:      score += 10
        if f["word_count"] < 4:        score += 25
        elif f["word_count"] < 8:      score += 10
        if f["avg_word_len"] > 8.0:    score += 35
        elif f["avg_word_len"] > 7.0:  score += 25
        elif f["avg_word_len"] > 6.5:  score += 15
        if f["phrase_hits"] >= 3:      score += 50
        elif f["phrase_hits"] == 2:    score += 35
        elif f["phrase_hits"] == 1:    score += 20
        if f["spam_hits"] >= 4:        score += 30
        elif f["spam_hits"] >= 2:      score += 15
        elif f["spam_hits"] == 1:      score += 8
        if f["superlative_count"] >= 3: score += 20
        elif f["superlative_count"] >= 1: score += 8
        if f["has_url"]:               score += 25
        if f["avg_sentence_len"] < 4:  score += 15
        if f["ru_phrase_hits"] >= 3:   score += 45
        elif f["ru_phrase_hits"] == 2: score += 30
        elif f["ru_phrase_hits"] == 1: score += 18
        if f["ru_spam_hits"] >= 4:     score += 28
        elif f["ru_spam_hits"] >= 2:   score += 14
        elif f["ru_spam_hits"] == 1:   score += 6
        return min(score, 100.0)

    def get_rule_score(self, features: dict) -> float:
        return self.__rule_score(features)


class FakeReviewDetector(BaseReviewDetector):
    def __init__(self):
        self.__classifier = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli"
        )
        self.__extractor = FeatureExtractor()

    def get_classifier(self):
        return self.__classifier

    def analyze(self, text: str):
        text = self.preprocess(text)
        result = self.__classifier(text, ["fake review", "genuine review"])
        scores   = dict(zip(result["labels"], result["scores"]))
        bert_prob = scores.get("fake review", 0.0) * 100
        features  = self.__extractor.extract(text)
        rule_score = self.__extractor.get_rule_score(features)
        combined   = 0.35 * bert_prob + 0.65 * rule_score
        verdict    = "Fake" if combined >= 30 else "Genuine"
        confidence = combined if verdict == "Fake" else (100 - combined)
        return verdict, round(confidence, 1), features, round(bert_prob, 1), round(rule_score, 1)

    def get_reasons(self, verdict, features, bert_prob, rule_score):
        r = []
        lang = "Roman Urdu/Pakistani English" if features["is_roman_urdu"] else "English"
        if verdict == "Fake":
            r.append((f"AI detected {bert_prob:.0f}% probability of fake review", "AI"))
            if features["excl_count"] >= 2:
                r.append((f"{features['excl_count']} excessive exclamation marks", "Rule"))
            if features["phrase_hits"] > 0:
                r.append((f"Found {features['phrase_hits']} English marketing phrase(s)", "Rule"))
            if features["ru_phrase_hits"] > 0:
                r.append((f"Found {features['ru_phrase_hits']} Roman Urdu fake phrase(s)", "Rule"))
            if features["spam_hits"] >= 2:
                r.append((f"{features['spam_hits']} spam adjectives detected", "Rule"))
            if features["ru_spam_hits"] >= 2:
                r.append((f"{features['ru_spam_hits']} Roman Urdu spam words detected", "Rule"))
            if features["avg_word_len"] > 6.5:
                r.append(("Unusually formal language pattern", "Rule"))
            if features["unique_ratio"] < 0.55:
                r.append((f"Low vocabulary diversity: {features['unique_ratio']*100:.0f}%", "Rule"))
            if features["is_roman_urdu"]:
                r.append((f"Language detected: {lang}", "Lang"))
        else:
            r.append((f"AI detected {100-bert_prob:.0f}% genuine probability", "AI"))
            r.append(("Natural writing structure found", "Rule"))
            r.append((f"Good vocabulary diversity: {features['unique_ratio']*100:.0f}%", "Rule"))
            if features["excl_count"] == 0:
                r.append(("No excessive punctuation detected", "Rule"))
            if features["is_roman_urdu"]:
                r.append((f"Language detected: {lang}", "Lang"))
        return r


# ─────────────────────────────────────────────
#  CSV HISTORY PERSISTENCE
# ─────────────────────────────────────────────
HISTORY_CSV = "review_history.csv"
CSV_COLS = ["timestamp", "verdict", "confidence", "bert_prob", "rule_score", "lang", "text"]

def load_history_csv():
    if not os.path.exists(HISTORY_CSV):
        return []
    try:
        df = pd.read_csv(HISTORY_CSV, dtype=str)
        records = df.to_dict("records")
        for r in records:
            r["confidence"] = float(r.get("confidence", 0))
            r["bert_prob"]  = float(r.get("bert_prob",  0))
            r["rule_score"] = float(r.get("rule_score", 0))
        return records
    except Exception:
        return []

def save_history_csv(history: list):
    try:
        with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(history)
    except Exception:
        pass

def append_history_csv(entry: dict):
    file_exists = os.path.exists(HISTORY_CSV)
    try:
        with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(entry)
    except Exception:
        pass


# ─────────────────────────────────────────────
#  LIME WORD HIGHLIGHTER
# ─────────────────────────────────────────────
def get_lime_word_weights(detector: FakeReviewDetector, text: str, num_samples: int = 200):
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        return None, "lime not installed. Run: pip install lime"

    classifier = detector.get_classifier()
    extractor  = FeatureExtractor()

    def predict_proba(texts):
        results = []
        for t in texts:
            try:
                res = classifier(t, ["fake review", "genuine review"])
                scores = dict(zip(res["labels"], res["scores"]))
                bert_prob = scores.get("fake review", 0.0) * 100
                feats = extractor.extract(t)
                rule_s = extractor.get_rule_score(feats)
                combined = min(0.35 * bert_prob + 0.65 * rule_s, 100.0)
                p_fake    = combined / 100.0
                p_genuine = 1.0 - p_fake
                results.append([p_genuine, p_fake])
            except Exception:
                results.append([0.5, 0.5])
        return np.array(results)

    explainer = LimeTextExplainer(class_names=["Genuine", "Fake"])
    try:
        exp = explainer.explain_instance(
            text,
            predict_proba,
            num_features=min(30, len(text.split())),
            num_samples=num_samples,
            labels=[1],
        )
        word_weights = exp.as_list(label=1)
        return word_weights, None
    except Exception as e:
        return None, str(e)


def build_highlighted_html(text: str, word_weights: list) -> str:
    if not word_weights:
        return f"<span>{text}</span>"
    weight_map = {}
    for word, w in word_weights:
        weight_map[word.lower().strip()] = w
    max_abs = max((abs(w) for _, w in word_weights), default=1e-6)
    if max_abs < 1e-8:
        max_abs = 1e-8
    tokens = re.findall(r'\S+|\s+', text)
    parts = []
    for token in tokens:
        if token.isspace():
            parts.append(token)
            continue
        clean = re.sub(r'[^\w]', '', token).lower()
        w = weight_map.get(clean, None)
        if w is None:
            parts.append(f'<span class="word-neutral" title="No LIME signal">{token}</span>')
        else:
            intensity = min(abs(w) / max_abs, 1.0)
            alpha = 0.18 + 0.62 * intensity
            if w > 0:
                r, g, b = 192, 82, 42
                label = f"Fake signal: +{w:.3f}"
                dot_class = "lime-dot-fake"
            else:
                r, g, b = 74, 103, 65
                label = f"Genuine signal: {w:.3f}"
                dot_class = "lime-dot-genuine"
            style = (
                f"background:rgba({r},{g},{b},{alpha:.2f});"
                f"color:{'#3D1A08' if w > 0 else '#1A2E18'};"
                f"border-radius:3px;padding:1px 4px;cursor:default;"
            )
            parts.append(
                f'<span class="word-highlighted {dot_class}" style="{style}" title="{label}">{token}</span>'
            )
    return "".join(parts)


# ─────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = load_history_csv()
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Review Guardian", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=Courier+Prime:wght@400;700&display=swap');

:root {
    --cream:        #F5F0E8;
    --cream-dark:   #EDE6D6;
    --cream-border: #D9CEB8;
    --ink:          #1C1A15;
    --ink-mid:      #3D3A32;
    --ink-light:    #6B6558;
    --ink-faint:    #9A9385;
    --terracotta:   #C0522A;
    --terra-light:  #F0DDD4;
    --terra-dark:   #8A3618;
    --sage:         #4A6741;
    --sage-light:   #D4E6D0;
    --sage-dark:    #2E4428;
    --sand:         #B8A882;
    --surface:      #FDFAF4;
    --surface-alt:  #F8F3E8;
}

*,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }

html,body,[data-testid="stAppViewContainer"],.stApp {
    background-color: var(--cream) !important;
    font-family: 'DM Sans', sans-serif !important;
    overflow-x: hidden !important;
}

#MainMenu,footer,header { visibility:hidden; }
[data-testid="stDecoration"] { display:none; }

@keyframes fadeUp    { from{opacity:0;transform:translateY(24px);} to{opacity:1;transform:translateY(0);} }
@keyframes fadeIn    { from{opacity:0;} to{opacity:1;} }
@keyframes slideLeft { from{opacity:0;transform:translateX(-20px);} to{opacity:1;transform:translateX(0);} }
@keyframes stampPop  { 0%{opacity:0;transform:scale(0.7) rotate(-8deg);} 70%{transform:scale(1.05) rotate(2deg);} 100%{opacity:1;transform:scale(1) rotate(0);} }
@keyframes navDrop   { from{opacity:0;transform:translateY(-100%);} to{opacity:1;transform:translateY(0);} }
@keyframes verdictIn { 0%{opacity:0;transform:scale(0.92) translateY(16px);} 60%{transform:scale(1.02) translateY(-2px);} 100%{opacity:1;transform:scale(1) translateY(0);} }
@keyframes floatBob  { 0%,100%{transform:translateY(0);} 50%{transform:translateY(-6px);} }
@keyframes spinRing  { from{transform:rotate(0deg);} to{transform:rotate(360deg);} }
@keyframes rowSlide  { from{opacity:0;transform:translateX(-10px);} to{opacity:1;transform:translateX(0);} }
@keyframes meterFill { from{width:0%;} to{width:var(--meter-pct);} }
@keyframes spinLoader{ from{transform:rotate(0deg);} to{transform:rotate(360deg);} }
@keyframes pulse     { 0%,100%{opacity:1;} 50%{opacity:0.4;} }

.navbar {
    position: sticky; top: 0; z-index: 999;
    background: rgba(245,240,232,0.96);
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--cream-border);
    padding: 0 3rem;
    display: flex; justify-content: space-between; align-items: center;
    height: 64px;
    animation: navDrop 0.6s cubic-bezier(0.22,1,0.36,1) both;
}
.nav-left { display:flex; align-items:center; gap:14px; }
.nav-logomark {
    width: 34px; height: 34px;
    background: var(--ink); border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; transition: background 0.25s, transform 0.3s cubic-bezier(0.34,1.56,0.64,1);
    cursor: default;
}
.nav-logomark:hover { background: var(--terracotta); transform: scale(1.12) rotate(-5deg); }
.nav-logo { font-family:'DM Serif Display',serif; font-size:1.25rem; color:var(--ink); letter-spacing:-0.3px; }
.nav-right { display:flex; align-items:center; gap:24px; }
.nav-link {
    font-family:'Courier Prime',monospace; font-size:0.6rem;
    letter-spacing:2px; text-transform:uppercase; color:var(--ink-faint);
    cursor:default; transition:color 0.2s; padding:4px 0;
    border-bottom: 1px solid transparent;
}
.nav-link:hover { color:var(--ink); border-bottom-color:var(--ink); }

.hero-wrap {
    max-width:1100px; margin:0 auto; padding:5.5rem 3rem 4.5rem;
    display:grid; grid-template-columns:1fr auto; gap:4rem; align-items:center;
}
.hero-eyebrow {
    display:flex; align-items:center; gap:12px; margin-bottom:1.4rem;
    font-family:'Courier Prime',monospace; font-size:0.6rem;
    letter-spacing:3px; text-transform:uppercase; color:var(--terracotta);
    animation: slideLeft 0.6s ease 0.2s both;
}
.hero-eyebrow::before { content:''; width:32px; height:1px; background:var(--terracotta); display:inline-block; }
.hero-title {
    font-family:'DM Serif Display',serif;
    font-size:clamp(2rem,5vw,4.6rem); line-height:1.05;
    color:var(--ink); letter-spacing:-2px;
    animation: fadeUp 0.8s ease 0.3s both;
}
.hero-title em { font-style:italic; color:var(--terracotta); }
.hero-body {
    font-size:0.97rem; color:var(--ink-light); line-height:1.9;
    font-weight:300; margin-top:1.2rem; max-width:460px;
    animation: fadeUp 0.7s ease 0.45s both;
}
.hero-badges { display:flex; flex-wrap:wrap; gap:8px; margin-top:1.8rem; animation:fadeUp 0.7s ease 0.6s both; }
.badge {
    background:var(--surface); border:1px solid var(--cream-border);
    border-radius:2px; padding:5px 14px;
    font-family:'Courier Prime',monospace; font-size:0.65rem;
    color:var(--ink-mid); letter-spacing:1px; text-transform:uppercase;
    transition:all 0.22s ease; cursor:default;
}
.badge:hover { background:var(--ink); color:var(--cream); border-color:var(--ink); transform:translateY(-2px); }
.badge.accent { background:var(--ink); color:var(--cream); border-color:var(--ink); }

.hero-stamp-wrap { animation: stampPop 0.9s cubic-bezier(0.34,1.56,0.64,1) 0.5s both; }
.hero-stamp {
    width:200px; height:200px; border:2px solid var(--ink);
    border-radius:50%; display:flex; align-items:center; justify-content:center;
    position:relative; cursor:default;
    transition: transform 0.4s cubic-bezier(0.34,1.56,0.64,1);
    animation: floatBob 5s ease-in-out 1.5s infinite;
}
.hero-stamp:hover { transform: scale(1.06) rotate(4deg); }
.hero-stamp::after {
    content:''; position:absolute; inset:-8px; border-radius:50%;
    border:1.5px dashed var(--terracotta); opacity:0;
    transition:opacity 0.3s; animation:spinRing 12s linear infinite;
}
.hero-stamp:hover::after { opacity:0.5; }
.stamp-inner {
    width:168px; height:168px; border:1px solid var(--cream-border);
    border-radius:50%; display:flex; flex-direction:column;
    align-items:center; justify-content:center; gap:8px;
    background:var(--surface);
}
.stamp-icon { font-size:2.5rem; line-height:1; transition:transform 0.3s cubic-bezier(0.34,1.56,0.64,1); }
.hero-stamp:hover .stamp-icon { transform:scale(1.18) rotate(-8deg); }
.stamp-text { font-family:'DM Serif Display',serif; font-size:1rem; color:var(--ink); text-align:center; line-height:1.3; }
.stamp-live {
    position:absolute; top:2px; right:12px;
    background:var(--terracotta); color:white;
    font-family:'Courier Prime',monospace; font-size:0.5rem;
    letter-spacing:1.5px; text-transform:uppercase;
    padding:3px 8px; border-radius:2px;
}

.stats-grid {
    max-width:1100px; margin:0 auto;
    display:grid; grid-template-columns:repeat(4,1fr);
    gap:1px; background:var(--cream-border);
    border:1px solid var(--cream-border); border-radius:6px; overflow:hidden;
    animation: fadeUp 0.6s ease 0.7s both;
}
.stat-cell { background:var(--surface); padding:24px 20px; text-align:center; }
.stat-num { font-family:'DM Serif Display',serif; font-size:2.6rem; line-height:1; letter-spacing:-1px; color:var(--ink); }
.stat-num.terra { color:var(--terracotta); }
.stat-num.sage  { color:var(--sage); }
.stat-label { font-family:'Courier Prime',monospace; font-size:0.52rem; letter-spacing:2px; text-transform:uppercase; color:var(--ink-faint); margin-top:5px; }

.section-label {
    display:flex; align-items:center; gap:10px;
    font-family:'Courier Prime',monospace; font-size:0.58rem;
    letter-spacing:3px; text-transform:uppercase; color:var(--ink-faint);
    margin-bottom:0.6rem;
}
.section-label::after { content:''; flex:1; height:1px; background:var(--cream-border); }
.section-title { font-family:'DM Serif Display',serif; font-size:1.75rem; color:var(--ink); letter-spacing:-0.5px; margin-bottom:0.3rem; }
.section-sub { font-size:0.88rem; color:var(--ink-light); font-weight:300; margin-bottom:1.8rem; }

.features-strip {
    max-width:1100px; margin:0 auto; padding:4.5rem 3rem;
    display:grid; grid-template-columns:repeat(3,1fr); gap:12px;
}
.feature-card {
    background:var(--surface); border:1px solid var(--cream-border);
    border-top:3px solid var(--sand);
    border-radius:4px; padding:2rem 1.8rem;
    transition:transform 0.3s ease, border-color 0.3s, box-shadow 0.3s;
}
.feature-card:nth-child(1) { border-top-color: var(--terracotta); }
.feature-card:nth-child(3) { border-top-color: var(--sage); }
.feature-card:hover { transform:translateY(-6px); box-shadow:0 12px 32px rgba(28,26,21,0.08); }
.feature-icon { font-size:1.7rem; margin-bottom:1rem; display:block; transition:transform 0.3s cubic-bezier(0.34,1.56,0.64,1); }
.feature-card:hover .feature-icon { transform:scale(1.2) rotate(-5deg); }
.feature-title { font-family:'DM Serif Display',serif; font-size:1.12rem; color:var(--ink); margin-bottom:0.5rem; }
.feature-body { font-size:0.84rem; color:var(--ink-light); line-height:1.75; font-weight:300; }

.content-wrap { max-width:900px; margin:0 auto; padding:3.5rem 3rem; }

.lang-badge {
    display:inline-flex; align-items:center; gap:8px;
    border:1px solid var(--cream-border); border-radius:2px;
    padding:5px 14px; font-family:'Courier Prime',monospace;
    font-size:0.62rem; color:var(--ink-mid); letter-spacing:1.5px;
    text-transform:uppercase; margin-bottom:1.2rem; background:var(--surface);
}
.lang-dot { width:6px; height:6px; border-radius:50%; background:var(--terracotta); }

.verdict {
    padding:1.8rem 2rem; margin:1.5rem 0;
    border:1px solid; border-radius:6px;
    display:grid; grid-template-columns:auto 1fr auto; gap:1.4rem; align-items:center;
    animation: verdictIn 0.6s cubic-bezier(0.34,1.56,0.64,1) both;
}
.verdict-fake    { background:#FBF0EB; border-color:#E8C4B4; }
.verdict-genuine { background:#EFF6ED; border-color:#C0D8BC; }
.verdict-icon-wrap {
    width:60px; height:60px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:1.7rem; flex-shrink:0;
    transition:transform 0.35s cubic-bezier(0.34,1.56,0.64,1);
}
.verdict-fake    .verdict-icon-wrap { background:var(--terra-light); border:1px solid #D9A898; }
.verdict-genuine .verdict-icon-wrap { background:var(--sage-light);  border:1px solid #A8C9A4; }
.verdict:hover .verdict-icon-wrap   { transform:scale(1.12) rotate(-5deg); }
.verdict-label { font-family:'Courier Prime',monospace; font-size:0.55rem; letter-spacing:2.5px; text-transform:uppercase; margin-bottom:3px; }
.verdict-fake    .verdict-label { color:var(--terra-dark); }
.verdict-genuine .verdict-label { color:var(--sage-dark); }
.verdict-heading { font-family:'DM Serif Display',serif; font-size:2rem; letter-spacing:-0.5px; line-height:1.1; }
.verdict-fake    .verdict-heading { color:var(--terracotta); }
.verdict-genuine .verdict-heading { color:var(--sage); }
.verdict-desc { font-size:0.83rem; color:var(--ink-light); font-weight:300; margin-top:3px; }
.verdict-scores { display:flex; flex-direction:column; gap:8px; flex-shrink:0; min-width:110px; }
.vscore-pill { background:rgba(255,255,255,0.65); border-radius:4px; padding:8px 14px; text-align:center; border:1px solid rgba(0,0,0,0.05); }
.vscore-label { font-family:'Courier Prime',monospace; font-size:0.5rem; letter-spacing:1.5px; text-transform:uppercase; color:var(--ink-faint); margin-bottom:2px; }
.vscore-val { font-family:'DM Serif Display',serif; font-size:1.6rem; line-height:1; letter-spacing:-0.5px; }

.conf-meter-wrap { margin: 1rem 0 1.5rem; animation: fadeUp 0.5s ease 0.3s both; }
.conf-meter-label { display:flex; justify-content:space-between; align-items:center; font-family:'Courier Prime',monospace; font-size:0.58rem; letter-spacing:2px; text-transform:uppercase; color:var(--ink-faint); margin-bottom:8px; }
.conf-meter-track { height: 8px; background: var(--cream-dark); border-radius: 4px; overflow: hidden; border: 1px solid var(--cream-border); }
.conf-meter-fill { height: 100%; border-radius: 4px; width: var(--meter-pct); animation: meterFill 1.1s cubic-bezier(0.22,1,0.36,1) 0.4s both; }
.conf-meter-fill.fake    { background: linear-gradient(90deg, #E8A882, #C0522A); }
.conf-meter-fill.genuine { background: linear-gradient(90deg, #A8C9A4, #4A6741); }
.conf-meter-ticks { display:flex; justify-content:space-between; font-family:'Courier Prime',monospace; font-size:0.48rem; color:var(--ink-faint); margin-top:4px; }

.loading-card { background: var(--surface); border: 1px solid var(--cream-border); border-radius: 6px; padding: 2.5rem; text-align: center; animation: fadeIn 0.3s ease both; }
.loading-spinner { width: 36px; height: 36px; border: 3px solid var(--cream-border); border-top-color: var(--terracotta); border-radius: 50%; animation: spinLoader 0.8s linear infinite; margin: 0 auto 1rem; }
.loading-text { font-family: 'Courier Prime', monospace; font-size: 0.72rem; letter-spacing: 2px; text-transform: uppercase; color: var(--ink-faint); animation: pulse 1.4s ease-in-out infinite; }
.loading-steps { display:flex; flex-direction:column; gap:8px; margin-top:1.2rem; text-align:left; max-width:280px; margin:1.2rem auto 0; }
.loading-step { display:flex; align-items:center; gap:10px; font-family:'Courier Prime',monospace; font-size:0.62rem; letter-spacing:1px; color:var(--ink-faint); }
.step-dot { width:6px; height:6px; border-radius:50%; background:var(--terracotta); animation: pulse 1s ease-in-out infinite; }

.insights-table { border:1px solid var(--cream-border); border-radius:6px; overflow:hidden; margin:1.5rem 0; animation:fadeUp 0.5s ease 0.2s both; }
.insights-table-header { padding:10px 16px; background:var(--cream-dark); font-family:'Courier Prime',monospace; font-size:0.58rem; letter-spacing:2.5px; text-transform:uppercase; color:var(--ink-faint); border-bottom:1px solid var(--cream-border); }
.insight-row-item { display:flex; align-items:center; justify-content:space-between; padding:11px 16px; border-bottom:1px solid var(--cream-border); background:var(--surface); font-size:0.86rem; color:var(--ink-mid); line-height:1.5; animation: rowSlide 0.4s ease both; }
.insight-row-item:last-child { border-bottom:none; }
.insight-row-item:hover { background:var(--cream); }
.insight-tag { font-family:'Courier Prime',monospace; font-size:0.5rem; letter-spacing:1px; text-transform:uppercase; padding:3px 9px; border-radius:2px; flex-shrink:0; margin-left:12px; }
.itag-ai   { background:var(--terra-light); color:var(--terra-dark); border:1px solid #E8C4B4; }
.itag-rule { background:var(--cream-dark); color:var(--ink-light); border:1px solid var(--cream-border); }
.itag-lang { background:var(--sage-light); color:var(--sage-dark); border:1px solid #C0D8BC; }

.lime-panel { background:var(--surface); border:1px solid var(--cream-border); border-radius:6px; padding:1.8rem; margin:1.5rem 0; }
.lime-panel-title { display:flex; align-items:center; gap:10px; font-family:'Courier Prime',monospace; font-size:0.58rem; letter-spacing:3px; text-transform:uppercase; color:var(--ink-faint); margin-bottom:5px; }
.lime-panel-title::after { content:''; flex:1; height:1px; background:var(--cream-border); }
.lime-panel-subtitle { font-size:0.82rem; color:var(--ink-light); font-weight:300; margin-bottom:1.2rem; }
.lime-text-body { font-family:'Courier Prime',monospace; font-size:1rem; line-height:2.2; letter-spacing:0.2px; color:var(--ink); background:var(--cream); border-radius:4px; padding:1.2rem 1.4rem; border:1px solid var(--cream-border); word-wrap:break-word; }
.word-highlighted { display:inline; transition:filter 0.2s; position:relative; }
.word-highlighted:hover { filter:brightness(0.85); }
.word-neutral { display:inline; color:var(--ink-light); }
.lime-legend { display:flex; gap:1.6rem; margin-top:1.1rem; align-items:center; flex-wrap:wrap; }
.lime-legend-item { display:flex; align-items:center; gap:7px; font-family:'Courier Prime',monospace; font-size:0.6rem; letter-spacing:1.5px; text-transform:uppercase; color:var(--ink-faint); }
.lime-swatch { width:26px; height:13px; border-radius:3px; }
.lime-swatch-fake    { background:rgba(192,82,42,0.55); }
.lime-swatch-genuine { background:rgba(74,103,65,0.55); }
.lime-swatch-neutral { background:transparent; border:1px solid var(--cream-border); }
.lime-top-words { display:flex; flex-wrap:wrap; gap:6px; margin-top:1rem; }
.lime-word-chip { display:inline-flex; align-items:center; gap:5px; padding:4px 12px; border-radius:3px; font-family:'Courier Prime',monospace; font-size:0.7rem; letter-spacing:0.5px; transition:transform 0.2s; cursor:default; }
.lime-word-chip:hover { transform:translateY(-2px); }
.lime-chip-fake    { background:rgba(192,82,42,0.12); color:var(--terra-dark); border:1px solid rgba(192,82,42,0.22); }
.lime-chip-genuine { background:rgba(74,103,65,0.12); color:var(--sage-dark);  border:1px solid rgba(74,103,65,0.22); }
.lime-chip-weight  { opacity:0.55; font-size:0.6rem; }
.lime-error-box { background:#FBF5E6; border:1px solid #E8D9B0; border-radius:4px; padding:1.2rem 1.6rem; font-size:0.84rem; color:var(--ink-mid); font-family:'Courier Prime',monospace; }

.history-panel { background: var(--surface); border: 1px solid var(--cream-border); border-radius: 6px; overflow: hidden; animation: fadeUp 0.4s ease both; }
.history-header { padding: 12px 18px; background: var(--cream-dark); border-bottom: 1px solid var(--cream-border); display:flex; justify-content:space-between; align-items:center; }
.history-title { font-family:'Courier Prime',monospace; font-size:0.58rem; letter-spacing:2.5px; text-transform:uppercase; color:var(--ink-faint); }
.history-empty { padding: 2rem; text-align:center; font-family:'Courier Prime',monospace; font-size:0.72rem; color:var(--ink-faint); letter-spacing:1px; }

.stTextArea textarea {
    background:var(--surface) !important; border:1px solid var(--cream-border) !important;
    border-radius:4px !important; color:var(--ink) !important;
    font-family:'Courier Prime',monospace !important; font-size:0.92rem !important;
    line-height:1.8 !important; padding:1.2rem !important;
    transition:border-color 0.25s, box-shadow 0.25s !important; box-shadow:none !important;
}
.stTextArea textarea:focus { border-color:var(--terracotta) !important; box-shadow:0 0 0 3px rgba(192,82,42,0.09) !important; }
.stTextArea textarea::placeholder { color:var(--ink-faint) !important; font-style:italic; }
.stSelectbox > div > div {
    background:var(--surface) !important; border:1px solid var(--cream-border) !important;
    border-radius:4px !important; color:var(--ink-mid) !important;
    font-family:'Courier Prime',monospace !important; font-size:0.82rem !important;
}
.stButton > button {
    background:var(--ink) !important; color:var(--cream) !important;
    font-family:'DM Sans',sans-serif !important; font-weight:600 !important;
    font-size:0.9rem !important; border:none !important; border-radius:3px !important;
    padding:0.9rem 2.5rem !important; transition:all 0.25s ease !important;
    width:100% !important; letter-spacing:0.3px !important;
}
.stButton > button:hover { background:var(--terracotta) !important; transform:translateY(-2px) !important; box-shadow:0 6px 20px rgba(192,82,42,0.22) !important; }
.stButton > button:active { transform:translateY(0) !important; }
.stDataFrame { border-radius:6px !important; border:1px solid var(--cream-border) !important; overflow:hidden !important; }
.stTabs [data-baseweb="tab-list"] { background:transparent !important; border-bottom:1px solid var(--cream-border) !important; padding:0 !important; gap:0 !important; max-width:900px; margin:0 auto; }
.stTabs [data-baseweb="tab"] { background:transparent !important; border:none !important; border-bottom:2px solid transparent !important; color:var(--ink-faint) !important; font-family:'Courier Prime',monospace !important; font-size:0.68rem !important; letter-spacing:1.5px !important; padding:1rem 2rem !important; text-transform:uppercase !important; border-radius:0 !important; transition:all 0.2s !important; }
.stTabs [data-baseweb="tab"]:hover { color:var(--ink) !important; }
.stTabs [aria-selected="true"] { background:transparent !important; color:var(--terracotta) !important; border-bottom:2px solid var(--terracotta) !important; }
[data-testid="metric-container"] { background:var(--surface) !important; border:1px solid var(--cream-border) !important; border-radius:6px !important; padding:1.4rem !important; box-shadow:none !important; transition:border-color 0.2s, transform 0.25s !important; }
[data-testid="metric-container"]:hover { border-color:var(--sand) !important; transform:translateY(-3px) !important; }
[data-testid="metric-container"] label { font-family:'Courier Prime',monospace !important; font-size:0.58rem !important; color:var(--ink-faint) !important; text-transform:uppercase !important; letter-spacing:2px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family:'DM Serif Display',serif !important; font-size:2.4rem !important; font-weight:400 !important; color:var(--ink) !important; }
.stWarning { background:#FBF5E6 !important; border:1px solid #E8D9B0 !important; border-radius:4px !important; }
.stSlider [data-baseweb="slider"] { padding:0 !important; }
.stSlider label { font-family:'Courier Prime',monospace !important; font-size:0.62rem !important; letter-spacing:1.5px !important; text-transform:uppercase !important; color:var(--ink-faint) !important; }

.site-footer { border-top:1px solid var(--cream-border); padding:2rem 3rem; display:flex; justify-content:space-between; align-items:center; max-width:1100px; margin:2rem auto 0; }
.footer-logo { font-family:'DM Serif Display',serif; font-size:1rem; color:var(--ink); }
.footer-copy { font-family:'Courier Prime',monospace; font-size:0.58rem; letter-spacing:1.5px; color:var(--ink-faint); text-transform:uppercase; }

::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--cream-dark); }
::-webkit-scrollbar-thumb { background:var(--sand); border-radius:3px; }

@media (max-width: 768px) {
    .navbar { padding: 0 1rem; height: 56px; }
    .nav-right { gap: 10px; }
    .nav-link { display: none; }
    .hero-wrap { grid-template-columns: 1fr; padding: 2.5rem 1.2rem 2rem; gap: 1.5rem; }
    .hero-stamp-wrap { display: none; }
    .hero-title { font-size: clamp(2rem, 8vw, 2.8rem); letter-spacing: -1px; }
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
    .features-strip { grid-template-columns: 1fr; padding: 2.5rem 1.2rem; gap: 10px; }
    .content-wrap { padding: 2rem 1.2rem; }
    .verdict { grid-template-columns: auto 1fr; grid-template-rows: auto auto; padding: 1.2rem; gap: 0.8rem; }
    .verdict-scores { grid-column: 1 / -1; flex-direction: row; min-width: unset; gap: 8px; }
    .vscore-pill { flex: 1; }
    .verdict-heading { font-size: 1.5rem; }
    .site-footer { flex-direction: column; gap: 8px; padding: 1.5rem 1.2rem; text-align: center; }
    [data-testid="block-container"] { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ── NAVBAR ──
st.markdown("""
<div class="navbar">
    <div class="nav-left">
        <div class="nav-logomark">🛡️</div>
        <div class="nav-logo">Review Guardian</div>
    </div>
    <div class="nav-right">
        <span class="nav-link">Analyse</span>
        <span class="nav-link">Dataset</span>
        <span class="nav-link">History</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── HERO ──
st.markdown("""
<div class="hero-wrap">
    <div class="hero-left">
        <div class="hero-eyebrow">AI-powered authenticity</div>
        <h1 class="hero-title">Spot the <em>fake</em><br>before it fools you.</h1>
        <p class="hero-body">
            Paste any review in English or Roman Urdu. Our hybrid engine — transformer AI
            plus 55+ linguistic signals — returns a verdict in seconds. LIME explainability
            shows <em>exactly which words</em> drove the decision.
        </p>
        <div class="hero-badges">
            <span class="badge">Google Reviews</span>
            <span class="badge">Amazon</span>
            <span class="badge">Trustpilot</span>
            <span class="badge">Daraz</span>
            <span class="badge accent">LIME XAI</span>
        </div>
    </div>
    <div class="hero-stamp-wrap">
        <div class="hero-stamp">
            <div class="stamp-live">Live</div>
            <div class="stamp-inner">
                <div class="stamp-icon">🛡️</div>
                <div class="stamp-text">Authenticity<br>Engine</div>
            </div>
        </div>
    </div>
</div>
<hr style="border:none;border-top:1px solid var(--cream-border);max-width:1100px;margin:0 auto;">
""", unsafe_allow_html=True)

st.markdown("""
<div style="max-width:1100px;margin:0 auto;padding:0 1.2rem;">
<div class="stats-grid">
    <div class="stat-cell"><div class="stat-num terra">55+</div><div class="stat-label">Fake signals</div></div>
    <div class="stat-cell"><div class="stat-num">2</div><div class="stat-label">Languages</div></div>
    <div class="stat-cell"><div class="stat-num">2K</div><div class="stat-label">Dataset rows</div></div>
    <div class="stat-cell"><div class="stat-num sage">XAI</div><div class="stat-label">LIME powered</div></div>
</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="features-strip">
    <div class="feature-card">
        <span class="feature-icon">🧠</span>
        <div class="feature-title">Transformer AI</div>
        <div class="feature-body">DistilBERT zero-shot classification on fake vs genuine — no training data needed for new domains.</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">📐</span>
        <div class="feature-title">Rule Engine</div>
        <div class="feature-body">55+ hand-crafted signals: exclamation marks, CAPS ratio, spam phrases, Roman Urdu markers, and more.</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">💡</span>
        <div class="feature-title">LIME Explainability</div>
        <div class="feature-body">Word-level highlighting shows exactly which tokens pushed the verdict — not just a black-box score.</div>
    </div>
</div>
<hr style="border:none;border-top:1px solid var(--cream-border);max-width:1100px;margin:0 auto;">
""", unsafe_allow_html=True)


# ── LOAD MODEL ──
if "detector" not in st.session_state:
    with st.spinner("Loading AI model..."):
        st.session_state.detector = FakeReviewDetector()
detector = st.session_state.detector


# ── TABS ──
tab1, tab2, tab3 = st.tabs(["✦  Analyse Review", "◈  Dataset", "🕐  History"])

# ════════════════════════════════════════════════════════
#  TAB 1: ANALYSE
# ════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-label">Input</div>
    <div class="section-title">Analyse any review</div>
    <div class="section-sub">Paste text in English or Roman Urdu / Pakistani English — language detection is automatic.</div>
    """, unsafe_allow_html=True)

    samples = {
        "Custom Input": "",
        "English — Bot Spam (Fake)":    "Best purchase ever! Buy now! 100% amazing quality!!! best best best!!!",
        "English — Genuine Review":     "The product is fine. Delivery was slow but it works as expected.",
        "English — Deepfake":           "I am incredibly impressed by the remarkable standard of this superior item. One must purchase immediately.",
        "English — Short Spam (Fake)":  "Great!!!",
        "English — Normal Review":      "Battery life is decent, lasts about 2 days. Build quality feels solid.",
        "Roman Urdu — Fake (Spam)":     "BEST PRODUCT EVER!!!! Bilkul lelo yaar!!! Zabardast quality!!! BUY NOW!!!",
        "Roman Urdu — Fake (Hype)":     "Masha Allah kya cheez hai!!! 100% amazing!!! Sab ko lena chahiye!!! ORDER ABHI!!!",
        "Roman Urdu — Genuine":         "Delivery thori late thi lekin product theek hai. Kaam ka hai. Price bhi theek tha.",
        "Roman Urdu — Genuine (Casual)":"Bhai sach mein acha laga. Koi jhooth nahi. Value for money hai.",
        "Roman Urdu — Mixed Fake":      "Yaar yaar yaar!!! Itna acha product!!! Best best best!!! Jaldi order karo limited stock hai!!!",
    }

    selected = st.selectbox("", list(samples.keys()), label_visibility="collapsed")
    default_text = samples[selected]
    text = st.text_area(
        "review", value=default_text, height=140,
        placeholder="Paste the full review text here — English ya Roman Urdu dono chalega...",
        label_visibility="collapsed"
    )

    lime_samples = st.slider(
        "LIME PERTURBATION SAMPLES — higher = more accurate, slower",
        min_value=50, max_value=500, value=150, step=50
    )

    run = st.button("Analyse Review →", key="run_btn")

    if run:
        if text.strip():
            loading_placeholder = st.empty()
            with loading_placeholder.container():
                st.markdown("""
                <div class="loading-card">
                    <div class="loading-spinner"></div>
                    <div class="loading-text">Analysing review…</div>
                    <div class="loading-steps">
                        <div class="loading-step"><div class="step-dot"></div>Preprocessing text</div>
                        <div class="loading-step"><div class="step-dot"></div>Running DistilBERT classifier</div>
                        <div class="loading-step"><div class="step-dot"></div>Scoring 55+ rule signals</div>
                        <div class="loading-step"><div class="step-dot"></div>Computing verdict</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            verdict, confidence, features, bert_prob, rule_score = detector.analyze(text)
            reasons = detector.get_reasons(verdict, features, bert_prob, rule_score)
            loading_placeholder.empty()

            lang_label = "Roman Urdu" if features["is_roman_urdu"] else "English"
            history_entry = {
                "timestamp": datetime.now().strftime("%H:%M · %d %b %Y"),
                "verdict": verdict,
                "confidence": confidence,
                "bert_prob": bert_prob,
                "rule_score": rule_score,
                "lang": lang_label,
                "text": text.strip(),
            }
            st.session_state.history.insert(0, history_entry)
            st.session_state.history = st.session_state.history[:50]
            append_history_csv(history_entry)
            st.session_state.last_result = {**history_entry, "features": features, "reasons": reasons}

            st.write("")

            lang_icon = "🇵🇰" if features["is_roman_urdu"] else "🌐"
            full_lang  = "Roman Urdu / Pakistani English" if features["is_roman_urdu"] else "English"
            st.markdown(
                f'<div class="lang-badge"><div class="lang-dot"></div>'
                f'{lang_icon}&nbsp;&nbsp;Language detected: {full_lang}</div>',
                unsafe_allow_html=True
            )

            conf_color = "#C0522A" if verdict == "Fake" else "#4A6741"

            if verdict == "Fake":
                st.markdown(f"""
                <div class="verdict verdict-fake">
                    <div class="verdict-icon-wrap">⛔</div>
                    <div>
                        <div class="verdict-label">Verdict</div>
                        <div class="verdict-heading">Likely Fake</div>
                        <div class="verdict-desc">High suspicion — this review shows markers of manufactured content.</div>
                    </div>
                    <div class="verdict-scores">
                        <div class="vscore-pill">
                            <div class="vscore-label">Confidence</div>
                            <div class="vscore-val" style="color:var(--terracotta)">{confidence:.0f}<span style="font-size:0.9rem;color:var(--ink-faint);">%</span></div>
                        </div>
                        <div class="vscore-pill">
                            <div class="vscore-label">Rule Score</div>
                            <div class="vscore-val" style="color:var(--sand)">{rule_score:.0f}<span style="font-size:0.9rem;color:var(--ink-faint);">/100</span></div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict verdict-genuine">
                    <div class="verdict-icon-wrap">✅</div>
                    <div>
                        <div class="verdict-label">Verdict</div>
                        <div class="verdict-heading">Likely Genuine</div>
                        <div class="verdict-desc">Low suspicion — this review appears to be authentic human writing.</div>
                    </div>
                    <div class="verdict-scores">
                        <div class="vscore-pill">
                            <div class="vscore-label">Confidence</div>
                            <div class="vscore-val" style="color:var(--sage)">{confidence:.0f}<span style="font-size:0.9rem;color:var(--ink-faint);">%</span></div>
                        </div>
                        <div class="vscore-pill">
                            <div class="vscore-label">Rule Score</div>
                            <div class="vscore-val" style="color:var(--sand)">{rule_score:.0f}<span style="font-size:0.9rem;color:var(--ink-faint);">/100</span></div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

            meter_class = "fake" if verdict == "Fake" else "genuine"
            meter_label = f"{'FAKE' if verdict == 'Fake' else 'GENUINE'} CONFIDENCE METER"
            st.markdown(f"""
            <div class="conf-meter-wrap">
                <div class="conf-meter-label">
                    <span>{meter_label}</span>
                    <span style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:{conf_color}">{confidence:.0f}%</span>
                </div>
                <div class="conf-meter-track">
                    <div class="conf-meter-fill {meter_class}" style="--meter-pct:{confidence:.0f}%"></div>
                </div>
                <div class="conf-meter-ticks"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div>
            </div>
            """, unsafe_allow_html=True)

            # ── INSIGHTS TABLE ──
            tag_map = {"AI": "itag-ai", "Rule": "itag-rule", "Lang": "itag-lang"}
            rows_html = ""
            for i, (reason_text, tag_type) in enumerate(reasons):
                delay = i * 60
                tag_class = tag_map.get(tag_type, "itag-rule")
                rows_html += (
                    f'<div class="insight-row-item" style="animation-delay:{delay}ms">'
                    f'<span>{reason_text}</span>'
                    f'<span class="insight-tag {tag_class}">{tag_type}</span></div>'
                )
            st.markdown(f"""
            <div class="insights-table">
                <div class="insights-table-header">Analysis Details</div>
                {rows_html}
            </div>""", unsafe_allow_html=True)

            # ── LIME ──
            st.markdown('<div class="section-label" style="margin-top:2rem">LIME Explainability</div>', unsafe_allow_html=True)
            lime_loading = st.empty()
            with lime_loading.container():
                st.markdown(f"""
                <div class="loading-card" style="padding:1.5rem">
                    <div class="loading-spinner" style="width:24px;height:24px;border-width:2px;margin-bottom:0.7rem"></div>
                    <div class="loading-text">Running LIME with {lime_samples} samples…</div>
                </div>
                """, unsafe_allow_html=True)

            word_weights, lime_error = get_lime_word_weights(detector, text.strip(), num_samples=lime_samples)
            lime_loading.empty()

            if lime_error:
                st.markdown(f'<div class="lime-error-box">⚠ LIME unavailable: {lime_error}<br>Install with: <code>pip install lime</code></div>', unsafe_allow_html=True)
            else:
                highlighted_html = build_highlighted_html(text.strip(), word_weights)
                fake_words    = sorted([(w, s) for w, s in word_weights if s > 0], key=lambda x: -x[1])[:8]
                genuine_words = sorted([(w, s) for w, s in word_weights if s < 0], key=lambda x: x[1])[:8]
                fake_chips    = "".join([f'<span class="lime-word-chip lime-chip-fake">{w} <span class="lime-chip-weight">+{s:.2f}</span></span>' for w, s in fake_words])
                genuine_chips = "".join([f'<span class="lime-word-chip lime-chip-genuine">{w} <span class="lime-chip-weight">{s:.2f}</span></span>' for w, s in genuine_words])

                st.markdown(f"""
                <div class="lime-panel">
                    <div class="lime-panel-title">Word-Level Influence Map</div>
                    <div class="lime-panel-subtitle">
                        Each word is coloured by its LIME contribution.
                        <strong style="color:#8A3618">Red</strong> words push toward <em>Fake</em>;
                        <strong style="color:#2E4428">green</strong> words push toward <em>Genuine</em>.
                        Hover a word to see its exact score.
                    </div>
                    <div class="lime-text-body">{highlighted_html}</div>
                    <div class="lime-legend">
                        <div class="lime-legend-item"><div class="lime-swatch lime-swatch-fake"></div>Fake signal</div>
                        <div class="lime-legend-item"><div class="lime-swatch lime-swatch-genuine"></div>Genuine signal</div>
                        <div class="lime-legend-item"><div class="lime-swatch lime-swatch-neutral"></div>Neutral</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if fake_words or genuine_words:
                    st.markdown('<div class="section-label" style="margin-top:1.5rem">Top Influential Words</div>', unsafe_allow_html=True)
                    col_fake, col_genuine = st.columns(2)
                    with col_fake:
                        st.markdown(f"""
                        <div style="font-family:'Courier Prime',monospace;font-size:0.58rem;letter-spacing:2px;text-transform:uppercase;color:#8A3618;margin-bottom:10px;">🔴 Fake Signals</div>
                        <div class="lime-top-words">{fake_chips if fake_chips else '<span style="color:var(--ink-faint);font-size:0.82rem;">None detected</span>'}</div>
                        """, unsafe_allow_html=True)
                    with col_genuine:
                        st.markdown(f"""
                        <div style="font-family:'Courier Prime',monospace;font-size:0.58rem;letter-spacing:2px;text-transform:uppercase;color:#2E4428;margin-bottom:10px;">🟢 Genuine Signals</div>
                        <div class="lime-top-words">{genuine_chips if genuine_chips else '<span style="color:var(--ink-faint);font-size:0.82rem;">None detected</span>'}</div>
                        """, unsafe_allow_html=True)

            # ── CHARTS ──
            st.markdown('<div class="section-label" style="margin-top:2rem">Visual Analysis</div>', unsafe_allow_html=True)
            import plotly.graph_objects as go

            fake_color    = "#C0522A"
            genuine_color = "#4A6741"
            primary_color = fake_color if verdict == "Fake" else genuine_color
            bg_color      = "#FDFAF4"
            grid_color    = "#D9CEB8"
            text_color    = "#1C1A15"
            faint_color   = "#9A9385"

            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    number={"suffix": "%", "font": {"size": 36, "color": primary_color, "family": "DM Serif Display, Georgia, serif"}},
                    title={"text": f"{'Fake' if verdict == 'Fake' else 'Genuine'} Confidence",
                           "font": {"size": 10, "color": faint_color, "family": "Courier Prime, monospace"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickfont": {"size": 9, "color": faint_color}, "tickcolor": grid_color},
                        "bar": {"color": primary_color, "thickness": 0.28},
                        "bgcolor": bg_color, "borderwidth": 0,
                        "steps": [
                            {"range": [0,  30], "color": "#EFF6ED"},
                            {"range": [30, 60], "color": "#FBF5E6"},
                            {"range": [60, 100],"color": "#FBF0EB"},
                        ],
                        "threshold": {"line": {"color": primary_color, "width": 3}, "thickness": 0.8, "value": confidence},
                    }
                ))
                gauge_fig.update_layout(height=260, margin=dict(t=60, b=20, l=30, r=30), paper_bgcolor=bg_color, font_color=text_color)
                st.plotly_chart(gauge_fig, use_container_width=True)

            with chart_col2:
                bar_fig = go.Figure()
                bar_fig.add_trace(go.Bar(
                    x=["AI Score (35%)", "Rule Score (65%)", "Combined"],
                    y=[bert_prob, rule_score, min(0.35*bert_prob + 0.65*rule_score, 100)],
                    marker_color=["#B8A882", "#8A7A5A", primary_color],
                    text=[f"{bert_prob:.0f}", f"{rule_score:.0f}", f"{min(0.35*bert_prob+0.65*rule_score,100):.0f}"],
                    textposition="outside",
                    textfont={"size": 12, "color": text_color, "family": "DM Serif Display, Georgia, serif"},
                    width=0.5,
                ))
                bar_fig.update_layout(
                    title={"text": "SCORE BREAKDOWN", "font": {"size": 10, "color": faint_color, "family": "Courier Prime, monospace"}, "x": 0},
                    height=260, margin=dict(t=50, b=20, l=10, r=10),
                    paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                    yaxis={"range": [0, 115], "showgrid": True, "gridcolor": grid_color, "tickfont": {"size": 9, "color": faint_color}, "zeroline": False},
                    xaxis={"tickfont": {"size": 10, "color": text_color, "family": "Courier Prime, monospace"}, "showgrid": False},
                    showlegend=False,
                )
                st.plotly_chart(bar_fig, use_container_width=True)

            if word_weights and not lime_error:
                top_n = sorted(word_weights, key=lambda x: abs(x[1]), reverse=True)[:15]
                top_n_sorted = sorted(top_n, key=lambda x: x[1])
                wds    = [w for w, _ in top_n_sorted]
                wts    = [s for _, s in top_n_sorted]
                colors = [fake_color if s > 0 else genuine_color for s in wts]
                lime_bar = go.Figure(go.Bar(
                    x=wts, y=wds, orientation='h',
                    marker_color=colors,
                    text=[f"{s:+.3f}" for s in wts],
                    textposition="outside",
                    textfont={"size": 10, "color": text_color, "family": "Courier Prime, monospace"},
                ))
                lime_bar.update_layout(
                    title={"text": "LIME WORD CONTRIBUTIONS (top 15)", "font": {"size": 10, "color": faint_color, "family": "Courier Prime, monospace"}, "x": 0},
                    height=max(320, len(top_n_sorted) * 28),
                    margin=dict(t=50, b=20, l=10, r=80),
                    paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                    xaxis={"showgrid": True, "gridcolor": grid_color, "zeroline": True, "zerolinecolor": grid_color,
                           "tickfont": {"size": 9, "color": faint_color}, "title": "LIME weight (positive = fake)"},
                    yaxis={"tickfont": {"size": 10, "color": text_color, "family": "Courier Prime, monospace"}, "showgrid": False},
                    showlegend=False,
                )
                st.plotly_chart(lime_bar, use_container_width=True)

            radar_labels = ["Exclamations","CAPS Ratio","Fake Phrases","Spam Words","RU Phrases","RU Spam","Low Vocab","Superlatives"]
            radar_vals = [
                min(features["excl_count"] / 6 * 100, 100),
                min(features["caps_ratio"] / 0.4 * 100, 100),
                min(features["phrase_hits"] / 4 * 100, 100),
                min(features["spam_hits"] / 6 * 100, 100),
                min(features["ru_phrase_hits"] / 4 * 100, 100),
                min(features["ru_spam_hits"] / 6 * 100, 100),
                max(0, (0.7 - features["unique_ratio"]) / 0.7 * 100),
                min(features["superlative_count"] / 5 * 100, 100),
            ]
            rv_closed = radar_vals + [radar_vals[0]]
            rl_closed = radar_labels + [radar_labels[0]]
            r_int = int(primary_color[1:3], 16)
            g_int = int(primary_color[3:5], 16)
            b_int = int(primary_color[5:7], 16)
            radar_fig = go.Figure()
            radar_fig.add_trace(go.Scatterpolar(
                r=rv_closed, theta=rl_closed, fill="toself",
                fillcolor=f"rgba({r_int},{g_int},{b_int},0.12)",
                line=dict(color=primary_color, width=2),
                marker=dict(color=primary_color, size=5),
            ))
            radar_fig.update_layout(
                polar=dict(
                    bgcolor=bg_color,
                    radialaxis=dict(visible=True, range=[0,100], tickfont={"size":8,"color":faint_color}, gridcolor=grid_color, linecolor=grid_color, tickvals=[25,50,75,100]),
                    angularaxis=dict(tickfont={"size":10,"color":text_color,"family":"Courier Prime, monospace"}, gridcolor=grid_color, linecolor=grid_color),
                ),
                title={"text": "FAKE SIGNAL RADAR", "font": {"size": 10, "color": faint_color, "family": "Courier Prime, monospace"}, "x": 0},
                height=340, margin=dict(t=50, b=20, l=40, r=40),
                paper_bgcolor=bg_color, showlegend=False,
            )
            st.plotly_chart(radar_fig, use_container_width=True)

            st.markdown('<div class="section-label" style="margin-top:1rem">Signal Breakdown</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([
                ["Word Count",              features["word_count"]],
                ["Exclamation Marks",       features["excl_count"]],
                ["CAPS Ratio",              f"{features['caps_ratio']*100:.0f}%"],
                ["Unique Words",            f"{features['unique_ratio']*100:.0f}%"],
                ["Repeated Words",          features["repetition"]],
                ["English Fake Phrases",    features["phrase_hits"]],
                ["Roman Urdu Fake Phrases", features["ru_phrase_hits"]],
                ["English Spam Adjectives", features["spam_hits"]],
                ["Roman Urdu Spam Words",   features["ru_spam_hits"]],
                ["Superlatives",            features["superlative_count"]],
                ["Roman Urdu Detected",     "Yes" if features["is_roman_urdu"] else "No"],
            ], columns=["Signal", "Value"]),
            hide_index=True, use_container_width=True)

        else:
            st.warning("Please paste a review to analyse.")

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  TAB 2: DATASET
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-label">Dataset</div>
    <div class="section-title">Review dataset overview</div>
    <div class="section-sub">1,000 English reviews + 1,000 Roman Urdu / Pakistani English reviews — 50/50 fake vs genuine split.</div>
    """, unsafe_allow_html=True)
    try:
        df = pd.read_csv("fakeReviewDataset1.csv")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Reviews",   f"{len(df):,}")
        c2.metric("Fake Reviews",    f"{(df['label']=='Fake').sum():,}")
        c3.metric("Genuine Reviews", f"{(df['label']=='Genuine').sum():,}")
        c4.metric("Languages", "2")
        st.write("")
        lang_filter = st.selectbox("Filter by language", ["All", "English (rows 1–1000)", "Roman Urdu (rows 1001–2000)"])
        if lang_filter == "English (rows 1–1000)":
            view_df = df[df['review_id'] <= 1000]
        elif lang_filter == "Roman Urdu (rows 1001–2000)":
            view_df = df[df['review_id'] > 1000]
        else:
            view_df = df
        st.dataframe(view_df.head(30), use_container_width=True)
    except Exception as e:
        st.error(f"Dataset file not found or error: {e}. Place fakeReviewDataset1.csv in the project root.")
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  TAB 3: HISTORY
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-label">Session + Persistent</div>
    <div class="section-title">Analysis History</div>
    <div class="section-sub">History persists across reloads via <code>review_history.csv</code>. Last 50 analyses shown.</div>
    """, unsafe_allow_html=True)

    history = st.session_state.history

    if history:
        col_dl, col_clear, _ = st.columns([1.5, 1.5, 4])
        with col_dl:
            hist_df = pd.DataFrame(history)[CSV_COLS]
            csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Export CSV",
                data=csv_bytes,
                file_name="review_history.csv",
                mime="text/csv",
                key="export_history_csv"
            )
        with col_clear:
            if st.button("🗑️ Clear History", key="clear_history"):
                st.session_state.history = []
                st.session_state.last_result = None
                if os.path.exists(HISTORY_CSV):
                    os.remove(HISTORY_CSV)
                st.rerun()

        st.write("")

        total_h   = len(history)
        fakes_h   = sum(1 for h in history if h["verdict"] == "Fake")
        genuine_h = total_h - fakes_h
        avg_conf  = sum(h["confidence"] for h in history) / total_h

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Analysed", total_h)
        s2.metric("Fake Detected",  fakes_h)
        s3.metric("Genuine",        genuine_h)
        s4.metric("Avg Confidence", f"{avg_conf:.0f}%")

        st.write("")
        st.markdown('<div class="section-label">Recent Analyses</div>', unsafe_allow_html=True)

        for h in history[:20]:
            with st.container():
                col_dot, col_info, col_score = st.columns([0.3, 6, 1.5])
                with col_dot:
                    dot = "🔴" if h["verdict"] == "Fake" else "🟢"
                    st.markdown(f"<div style='font-size:1.1rem;padding-top:4px'>{dot}</div>", unsafe_allow_html=True)
                with col_info:
                    preview = h["text"][:90] + "…" if len(h["text"]) > 90 else h["text"]
                    st.markdown(
                        f"<div style='font-family:Courier Prime,monospace;font-size:0.78rem;color:#3D3A32;line-height:1.5'>{preview}</div>"
                        f"<div style='font-family:Courier Prime,monospace;font-size:0.52rem;color:#9A9385;margin-top:2px'>"
                        f"{h['timestamp']} · {h['lang']}</div>",
                        unsafe_allow_html=True
                    )
                with col_score:
                    color = "#C0522A" if h["verdict"] == "Fake" else "#4A6741"
                    st.markdown(
                        f"<div style='text-align:right'>"
                        f"<div style='font-family:DM Serif Display,serif;font-size:1.4rem;color:{color};line-height:1'>{h['confidence']:.0f}%</div>"
                        f"<div style='font-family:Courier Prime,monospace;font-size:0.5rem;color:#9A9385;text-transform:uppercase'>{h['verdict']}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                st.divider()
    else:
        st.markdown("""
        <div class="history-panel">
            <div class="history-empty">
                🕐 No analyses yet.<br>
                <span style="font-size:0.62rem;opacity:0.6">Run your first review in the Analyse tab.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── FOOTER ──
st.markdown("""
<div class="site-footer">
    <div class="footer-logo">Review Guardian</div>
    <div class="footer-copy">&copy; 2026 &middot; English + Roman Urdu &middot; LIME XAI</div>
</div>
""", unsafe_allow_html=True)