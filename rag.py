import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import difflib

DEFAULT_DB_DIR = "qa_vector_db"
DEFAULT_MODEL_PATH = "medical_embedding_model"

class MedicalRAGEngine:
    def __init__(self, db_dir=DEFAULT_DB_DIR, model_path=DEFAULT_MODEL_PATH):
        self.db_dir = db_dir
        self.model_path = model_path
        self.vectorstore = None
        self.embedding_model = None
        self._load_resources()

    def _load_resources(self):
        print(f"[LOAD] Loading RAG Engine from: {self.model_path}...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Embedding model not found at {self.model_path}")
        
        if not os.path.exists(self.db_dir):
            raise FileNotFoundError(f"Vector DB not found at {self.db_dir}. Run build_db.py first.")

        try:
            model_kwargs = {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'trust_remote_code': True
            }
            encode_kwargs = {'normalize_embeddings': True}
            
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.model_path,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )

            self.vectorstore = Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embedding_model
            )
            print(f"[OK] RAG Engine Ready ({self.vectorstore._collection.count()} vectors loaded).")
            
        except Exception as e:
            print(f"[ERROR] Critical Error loading RAG Engine: {e}")
            raise e

    @staticmethod
    def clean_merge(text1, text2, overlap_size=200):

        end_text1 = text1[-overlap_size:] 
        start_text2 = text2[:overlap_size]

        matcher = difflib.SequenceMatcher(None, end_text1, start_text2)
        match = matcher.find_longest_match(0, len(end_text1), 0, len(start_text2))
        
        if match.size > 30:
            return text1 + text2[match.b + match.size:]

        return text1 + " " + text2

    def search(self, query, k=4, qa_threshold=0.33, knowledge_threshold=0.5):

        if not self.vectorstore:
            return "System Error: Database not loaded."

        results_with_scores = self.vectorstore.similarity_search_with_score(query, k=k*2)
        
        final_context = []
        processed_sections = set()
        
        # Track counts to ensure balanced retrieval
        qa_count = 0
        knowledge_count = 0
        max_qa = max(2, k // 2)  # At most half should be QA
        min_knowledge = max(2, k // 2)  # At least half should be Knowledge

        for doc, score in results_with_scores:
            if len(final_context) >= k:
                if knowledge_count < min_knowledge and len(results_with_scores) > results_with_scores.index((doc, score)):
                    pass
                else:
                    break
            
            meta = doc.metadata
            dtype = meta.get('type', 'Unknown')
            source = meta.get('source', 'Unknown')
            
            # Apply type-specific threshold
            if dtype == 'QA':
                threshold = qa_threshold
                if qa_count >= max_qa:
                    continue
            elif dtype == 'Knowledge':
                threshold = knowledge_threshold
            else:
                threshold = knowledge_threshold
            
            # Threshold check with type-specific limit
            if score > threshold:
                continue
            
            # --- CASE A: QA PAIR ---
            if dtype == 'QA':
                content = (
                    f"--- CẶP HỎI ĐÁP ---\n"
                    f"Nguồn: {source}\n"
                    f"Câu hỏi: {doc.page_content}\n"
                    f"Câu trả lời: {meta.get('answer')}"
                )
                final_context.append(content)
                qa_count += 1
                
            # --- CASE B: KNOWLEDGE ARTICLE ---
            elif dtype == 'Knowledge':
                source = meta.get('source', 'Unknown')
                title = meta.get('title')
                section = meta.get('section')
                
                key = f"{source}_{title}_{section}"
                
                if key not in processed_sections:
                    total_chunks = meta.get('total_chunks', 1)
                    
                    # Sibling Reconstruction Logic
                    if total_chunks > 1:
                        # 1. Fetch ALL parts of this section (Chunk 0, 1, 2...)
                        full_section_docs = self.vectorstore.get(
                            where={"$and": [{"source": source}, {"title": title}, {"section": section}]}
                        )
                        
                        # 2. Sort them by index to ensure correct reading order
                        sorted_docs = sorted(
                            zip(full_section_docs['documents'], full_section_docs['metadatas']),
                            key=lambda x: x[1].get('chunk_index', 0)
                        )
                        
                        # 3. Clean Headers
                        cleaned_texts = []
                        # Define the header pattern added in build_db.py
                        header_prefix = f"Bài viết: {title} | Mục: {section}\nNội dung: "
                        
                        for i, (text_val, _) in enumerate(sorted_docs):
                            if i == 0:
                                # Keep header for the first chunk
                                cleaned_texts.append(text_val)
                            else:
                                # Remove header for subsequent chunks to avoid repetition
                                cleaned_texts.append(text_val.replace(header_prefix, ""))
                                
                        full_text = cleaned_texts[0]
                        for i in range(1, len(cleaned_texts)):
                            full_text = self.clean_merge(full_text, cleaned_texts[i])
                    else:
                        full_text = doc.page_content

                    # Format with Citation Source
                    content = (
                        f"--- TÀI LIỆU Y KHOA ---\n"
                        f"Nguồn: {source}\n"
                        f"Bài viết: {title}\n"
                        f"Mục: {section}\n"
                        f"Nội dung:\n{full_text}"
                    )
                    
                    final_context.append(content)
                    processed_sections.add(key)
                    knowledge_count += 1

        if not final_context:
            return "Không tìm thấy thông tin y tế phù hợp (Độ tin cậy thấp)."
        
        # Log the retrieval breakdown for debugging
        print(f"[RETRIEVAL] Retrieved {qa_count} QA pairs + {knowledge_count} Knowledge articles = {len(final_context)} total")
            
        return "\n\n".join(final_context)