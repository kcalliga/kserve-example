# KServe PythonModel for "Paragraph Chatbot" (TF‑IDF retrieval + FLAN‑T5 Small)
# Accepts JSON: {"question":"..."} or KServe v2-style {"inputs":[...]}; returns {"answer","context","similarity"}.
import os, re
from typing import List, Dict, Any
from collections import Counter
import numpy as np, pandas as pd
from kserve import Model, ModelServer

HF_MODEL = os.getenv("HF_MODEL_NAME", "google/flan-t5-small")
HF_FALLBACK = os.getenv("HF_FALLBACK_NAME", "t5-small")
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "64"))
HF_NUM_BEAMS = int(os.getenv("HF_NUM_BEAMS", "1"))
PARAGRAPH_PATH = os.getenv("PARAGRAPH_PATH", "paragraph.txt")
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.05"))

def sent_tokenize(text: str): return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
def word_tokenize(text: str): return re.findall(r"[a-z']+", text.lower())

def build_tfidf(sentences: List[str]):
    docs=[word_tokenize(s) for s in sentences]
    vocab=sorted({w for d in docs for w in d}); vidx={w:i for i,w in enumerate(vocab)}
    df=np.zeros(len(vocab))
    for toks in docs:
        for w in set(toks): df[vidx[w]]+=1
    idf=np.log((len(sentences)+1)/(df+1))+1
    rows=[]
    for i,toks in enumerate(docs):
        tf=np.zeros(len(vocab)); L=max(1,len(toks)); c=Counter(toks)
        for w,n in c.items(): tf[vidx[w]]=n/L
        vec=tf*idf; rows.append({"chunk_id":i,"chunk":sentences[i],"tfidf":vec})
    import pandas as pd
    return pd.DataFrame(rows), vidx, idf

def vectorize_query(q: str, vidx: Dict[str,int], idf):
    toks=word_tokenize(q); tf=np.zeros(len(vidx)); L=max(1,len(toks)); c=Counter(toks)
    for w,n in c.items():
        if w in vidx: tf[vidx[w]]=n/L
    return tf*idf

def cosine(a,b):
    na,nb=np.linalg.norm(a),np.linalg.norm(b)
    return 0.0 if na==0 or nb==0 else float(np.dot(a,b)/(na*nb))

class ParagraphChatbot(Model):
    def __init__(self, name:str):
        super().__init__(name)
        self.ready=False; self.vecdb=None; self.vidx=None; self.idf=None
        self.generator=None; self.tokenizer=None

    def load(self):
        if not os.path.exists(PARAGRAPH_PATH):
            raise RuntimeError(f"Paragraph file not found: {PARAGRAPH_PATH}")
        with open(PARAGRAPH_PATH,"r",encoding="utf-8") as f:
            paragraph=f.read().strip()
        sentences=sent_tokenize(paragraph)
        self.vecdb,self.vidx,self.idf=build_tfidf(sentences)

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        model_name=HF_MODEL
        try:
            tok=AutoTokenizer.from_pretrained(model_name)
            mdl=AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception:
            tok=AutoTokenizer.from_pretrained(HF_FALLBACK)
            mdl=AutoModelForSeq2SeqLM.from_pretrained(HF_FALLBACK)
            model_name=HF_FALLBACK
        self.tokenizer=tok
        self.generator=pipeline("text2text-generation", model=mdl, tokenizer=tok)
        self.ready=True
        return self.ready

    def _retrieve(self, question:str):
        qv=vectorize_query(question,self.vidx,self.idf)
        scored=[(cosine(qv,row["tfidf"]), row["chunk"]) for _,row in self.vecdb.iterrows()]
        scored.sort(reverse=True, key=lambda x:x[0])
        return scored[0] if scored else (0.0, "")

    def _generate(self, question:str, context:str)->str:
        prompt=(
            "Answer the question using only the context.\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            "Answer concisely:"
        )
        out=self.generator(prompt, max_new_tokens=HF_MAX_NEW_TOKENS, num_beams=HF_NUM_BEAMS, do_sample=False)[0]["generated_text"]
        return out.strip()

    def predict(self, request: Dict[str,Any])->Dict[str,Any]:
        if "question" in request:
            question=request["question"]
        elif "inputs" in request and isinstance(request["inputs"],list) and request["inputs"]:
            data=request["inputs"][0].get("data") or request["inputs"][0].get("contents")
            if isinstance(data,list) and data: question=data[0]
            else: raise ValueError("Unsupported 'inputs' format; expected list with 'data'.")
        else:
            raise ValueError("Missing 'question' in request.")
        sim,context=self._retrieve(question)
        if sim < SIM_THRESHOLD or not context:
            return {"answer":"I don’t have enough information in my notes to answer that.","context":"","similarity":sim}
        ans=self._generate(question, context)
        return {"answer":ans, "context":context, "similarity":sim}

if __name__=="__main__":
    ModelServer().start(models=[ParagraphChatbot("paragraph-chatbot")])
