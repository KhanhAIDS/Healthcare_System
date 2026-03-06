import json
import os
import pickle
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class MedicalNER:
    def __init__(self, model_path: str):
        print(f"[NER] Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer, 
                           aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)
    
    def extract(self, text: str, threshold: float = 0.5) -> Dict[str, List[str]]:
        """
        Extract entities grouped by 5 types: 
        disease, symptom, treatment, cause, diagnosis.
        """
        entities = {
            'disease': [], 
            'symptom': [], 
            'treatment': [],
            'cause': [], 
            'diagnosis': []
        }
        
        results = self.nlp(text)
        
        for e in results:
            if e['score'] > threshold:
                word = e['word'].strip().lower()

                if len(word) <= 2:
                    continue
                
                label = e['entity_group'].lower()
                
                # --- 5 LABEL MAPPING ---
                if 'ten_benh' in label:
                    entities['disease'].append(word)
                elif 'trieu_chung' in label:
                    entities['symptom'].append(word)
                elif 'dieu_tri' in label:
                    entities['treatment'].append(word)
                elif 'nguyen_nhan' in label:
                    entities['cause'].append(word)
                elif 'chan_doan' in label:
                    entities['diagnosis'].append(word)
                    
        # Remove duplicates
        return {k: list(set(v)) for k, v in entities.items()}


class ArticleKB:

    def __init__(self, rag_engine=None, ner_engine=None, index_path="kb_index.pkl"):
        self.index_path = index_path
        self.index = defaultdict(list)  # keyword -> list of chunk_ids
        self.articles = {}              # chunk_id -> metadata

        if os.path.exists(self.index_path):
            print(f"[KB] Found existing index at '{self.index_path}'. Loading...")
            self.load()
        elif rag_engine and ner_engine:
            print("[KB] No index found. Building from RAG Database (this may take a while)...")
            self.build_index(rag_engine, ner_engine)
            self.save()
        else:
            print("[KB] ⚠️ Warning: No index file and no RAG engine provided. KB is empty.")

    def save(self):
        try:
            with open(self.index_path, 'wb') as f:
                pickle.dump({'index': dict(self.index), 'articles': self.articles}, f)
            print(f"[KB] Saved index to {self.index_path}")
        except Exception as e:
            print(f"[KB] ❌ Error saving index: {e}")

    def load(self):
        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.index = data['index']
                self.articles = data['articles']
            self._normalize_links()
            print(f"[KB] Loaded {len(self.articles)} knowledge chunks.")
        except Exception as e:
            print(f"[KB] ❌ Error loading index: {e}")

    def _normalize_links(self):
        for _, meta in list(self.articles.items()):
            link_val = meta.get('link') or meta.get('source') or meta.get('url')
            if not link_val or link_val in {"#", "N/A"}:
                continue
            meta['link'] = link_val

    def build_index(self, rag, ner):
        docs = rag.vectorstore.get()['documents']
        metadatas = rag.vectorstore.get()['metadatas']

        count = 0
        skipped = 0
        print(f"[KB] Scanning {len(docs)} documents...")

        for i, (doc_text, meta) in enumerate(zip(docs, metadatas)):
            doc_type = meta.get('type', '').lower()
            if 'knowledge' not in doc_type:
                skipped += 1
                continue

            source = meta.get('source')
            if not source:
                skipped += 1
                continue

            extracted = ner.extract(doc_text)

            self.articles[i] = {
                'link': source,
                'title': meta.get('title', 'Unknown'),
                'section': meta.get('section', 'General'),
                'category': meta.get('category', ''),
                'chunk_index': meta.get('chunk_index', 0),
                'total_chunks': meta.get('total_chunks', 1),
            }

            has_entity = False
            for group in extracted.values():
                for word in group:
                    self.index[word].append(i)
                    has_entity = True

            if has_entity:
                count += 1

            if i % 500 == 0:
                print(f"   Processed {i} chunks...")

        print(f"[KB] Index built. Indexed {count} chunks. Skipped {skipped} (non-knowledge/missing source).")

    def search(self, query_entities: List[str], top_k: int = 5) -> List[Dict]:
        link_scores = Counter()
        link_metadata = {}

        for entity in query_entities:
            if entity not in self.index:
                continue

            for chunk_id in self.index[entity]:
                article = self.articles.get(chunk_id)
                if not article:
                    continue

                link = article.get('link')
                if link == "#":
                    raise ValueError(f"❌ CRITICAL: Article chunk {chunk_id} has invalid link '#'. "
                                   f"Metadata: {article}. Aborting search.")
                if not link or link in {"N/A"}:
                    continue

                link_scores[link] += 1
                if link not in link_metadata:
                    link_metadata[link] = article

        results = []
        for link, score in link_scores.most_common(top_k):
            meta = link_metadata[link]
            results.append({
                'link': link,
                'title': meta.get('title', ''),
                'section': meta.get('section', ''),
                'category': meta.get('category', ''),
                'score': score,
            })

        return results

class ProductKB:
    def __init__(self, db_dir="product_vector_db", embedding_model=None, score_threshold: float = 0.5):
        self.db_dir = db_dir
        self.db = None
        self.score_threshold = score_threshold
        
        if os.path.exists(self.db_dir):
            print(f"[ProductKB] Loading Product Database from '{self.db_dir}'...")
            try:
                # Use shared model if provided, otherwise load new one (Fallback)
                if embedding_model:
                    embedding = embedding_model
                    print("[ProductKB] Using Shared Embedding Model (RAM Optimized) ✅")
                else:
                    print("[ProductKB] Loading new Embedding Model (High RAM usage) ⚠️")
                    embedding = HuggingFaceEmbeddings(
                        model_name="medical_embedding_model",
                        model_kwargs={'device': 'cpu', 'trust_remote_code': True},
                        encode_kwargs={'normalize_embeddings': True}
                    )

                self.db = Chroma(persist_directory=self.db_dir, embedding_function=embedding)
                print("[ProductKB] Ready.")
            except Exception as e:
                print(f"[ProductKB] ❌ Error loading ChromaDB: {e}")
        else:
            print(f"[ProductKB] ⚠️ Warning: Database folder '{self.db_dir}' not found.")

    def search(self, query_text: str, top_k: int = 3) -> List[Dict]:
        if not self.db or not query_text.strip():
            return []
            
        try:
            # Semantic Search
            docs = self.db.similarity_search_with_score(query_text, k=top_k)
            
            results = []
            for doc, score in docs:
                # Chroma returns lower distance for better matches; filter weak ones
                if score is None or score > self.score_threshold:
                    continue
                meta = doc.metadata
                results.append({
                    'type': 'product',
                    'name': meta.get('name'),
                    'url': meta.get('url'),
                    'image': meta.get('image'),
                    'rating': meta.get('rating'),
                    'desc': f"Category: {meta.get('category')}"
                })
            return results
        except Exception as e:
            print(f"[ProductKB] Search Error: {e}")
            return []

class Recommender:
    # Add embedding_model argument to __init__
    def __init__(self, ner_path: str, embedding_model=None):
        print("="*50)
        print("📚 Initializing Recommender System")
        print("="*50)
        
        self.ner = MedicalNER(ner_path)
        # Import RAG engine to build article KB if needed
        from rag import MedicalRAGEngine
        try:
            rag_engine = MedicalRAGEngine()
        except Exception as e:
            print(f"[Recommender] Warning: Could not load RAG engine: {e}")
            rag_engine = None
        
        # Pass both rag_engine and ner_engine so KB can be built if needed
        self.article_kb = ArticleKB(rag_engine=rag_engine, ner_engine=self.ner, index_path="kb_index.pkl")
        
        # Pass embedding model to ProductKB
        self.product_kb = ProductKB(db_dir="product_vector_db", embedding_model=embedding_model)

        print("✅ Recommender Ready")
    
    # Remove user_id argument
    def recommend(self, query: str, top_k: int = 3) -> Dict:
        """
        Get recommendations based on current query.
        """
        # 1. Extract Keywords
        entities_dict = self.ner.extract(query)
        all_entities = []
        for elist in entities_dict.values():
            all_entities.extend(elist)
        
        # 2. Search Articles (only if entities found)
        articles = self.article_kb.search(all_entities, top_k=top_k) if all_entities else []
        
        # 3. Search Products (always use raw query for better semantic matching)
        # Products are general items (masks, tools) not medical entities
        products = self.product_kb.search(query, top_k=top_k)
        
        return {
            'articles': articles,
            'products': products
        }

# --- Testing Block ---
if __name__ == "__main__":
    # Ensure you have "NER_trained_model" folder in the same directory
    try:
        rec = Recommender("NER_trained_model")
        
        test_query = "Nguyên nhân gây ra ung thư phổi và cách chẩn đoán"
        print(f"\n🧪 Testing Query: '{test_query}'")
        
        result = rec.recommend(test_query, top_k=3)
        
        print("\n📋 Recommendations:")
        if not result.get('articles'):
            print("   No article recommendations found.")
        else:
            print("   Articles:")
            for r in result['articles']:
                print(f"   ⭐ Score: {r['score']} | Link: {r['link']}")
                print(f"      Title: {r['title']} | Section: {r['section']}")
        
        if result.get('products'):
            print("\n   Products:")
            for p in result['products']:
                print(f"   🛍️ {p['name']} | {p['rating']}")
                print(f"      {p['url']}")
                
    except Exception as e:
        print(f"❌ Test Failed: {e}")