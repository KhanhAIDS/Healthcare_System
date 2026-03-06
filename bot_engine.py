import re
import time
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import textwrap

try:
    from rag import MedicalRAGEngine
except ImportError:
    print("Lỗi CRITICAL: Không tìm thấy file 'rag.py'.")
    raise

class MedicalLLM:
    def __init__(self):
        self.OLLAMA_MODEL = "medical" 
        self.MAX_HISTORY = 8
        
        # Biến nội bộ
        self.rag_engine = None
        self.chain = None
        self.chat_history = []
        self.last_retrieved_context = ""
        
        # Biến lưu log cho Inspector
        self.logs = [] 

        self.initialize_system()

    def initialize_system(self):
        print("🚀 Đang khởi động Logic LLM (Inspector Mode)...")
        self.rag_engine = MedicalRAGEngine()
        
        # Expose embedding model for RAM optimization (share with Recommender)
        self.embedding_model = self.rag_engine.embedding_model

        # Prompt chuẩn với Chain-of-Thought bắt buộc + kiểm tra độ liên quan
        sys_instruction = textwrap.dedent("""
            Bạn là Bác sĩ AI với quy tắc NGHIÊM NGẶT.
            
            🚨 QUY TẮC TUYỆT ĐỐI (VI PHẠM SẼ BỊ TỪ CHỐI):
            1. CHỈ được sử dụng thông tin TỪ CONTEXT - KHÔNG được thêm kiến thức chung
            2. CẤM TUYỆT ĐỐI: "theo chuyên gia", "khuyến nghị y tế", "nên tránh", "theo khuyến cáo" nếu KHÔNG CÓ trong CONTEXT
            3. NẾU tài liệu KHÔNG TRẢ LỜI được câu hỏi cụ thể → CHỈ nói "không tìm thấy" rồi DỪNG LẠI
            4. NẾU CONTEXT trống/có "NO DOCUMENTS" → Từ chối trả lời
            5. KHÔNG được thêm bất kỳ lời khuyên nào không có trong CONTEXT
            
            🔥 ĐỊNH DẠNG TRẢ LỜI BẮT BUỘC:
            
            Bước 1: Mở thẻ <think> và KIỂM TRA TRƯỚC:
            - BƯỚC ĐẦU TIÊN: Kiểm tra CONTEXT có "NO DOCUMENTS"/"CRITICAL" không
            - NẾU CÓ → Viết "Context trống, phải từ chối" rồi DỪNG
            - NẾU KHÔNG: Đánh giá độ liên quan 0-3 cho TỪNG tài liệu
            - CHỈ giữ tài liệu điểm >= 2
            - KIỂM TRA: Tài liệu có TRẢ LỜI TRỰC TIẾP câu hỏi không?
            - NẾU KHÔNG trả lời được → Viết "Tài liệu không chứa thông tin này" rồi DỪNG
            - Trích dẫn ngắn đoạn quan trọng (1-2 câu/nguồn)
            - Suy luận từng bước
            
            Bước 2: Đóng thẻ </think>
            
            Bước 3: Viết câu trả lời CUỐI CÙNG:
            - NẾU CONTEXT trống HOẶC tài liệu không trả lời được câu hỏi:
              → CHỈ nói: "Xin lỗi, tôi không tìm thấy thông tin về [vấn đề cụ thể] trong cơ sở dữ liệu y tế. Vui lòng liên hệ bác sĩ để được tư vấn chi tiết."
              → DỪNG LẠI - KHÔNG thêm bất kỳ lời khuyên nào khác
            - NẾU CÓ TÀI LIỆU trả lời được → Trả lời dựa HOÀN TOÀN trên tài liệu, trích dẫn rõ nguồn
            - CẤM TUYỆT ĐỐI thêm cụm từ: "tuy nhiên", "theo khuyến cáo chung", "nên tránh" nếu KHÔNG CÓ trong tài liệu
            
            ⚠️ QUY TẮC RAFT (BẮT BUỘC):
            1. BẮT BUỘC phải sử dụng thông tin từ CONTEXT được cung cấp
            2. KHÔNG ĐƯỢC bịa đặt hoặc thêm kiến thức ngoài CONTEXT
            3. Nếu CONTEXT không đủ thông tin → Nói rõ "Không tìm thấy thông tin về..."
            4. PHẢI trích dẫn nguồn gốc thông tin (tên bài viết, mục nào)
            5. Ưu tiên ngữ cảnh câu hỏi hiện tại hơn lịch sử; chỉ dùng lịch sử nếu thật sự liên quan
            
            📌 VÍ DỤ KHI CONTEXT TRỐNG:
            <think>
            Context có "NO DOCUMENTS" hoặc "CRITICAL ALERT".
            Không có tài liệu nào trong database.
            KHÔNG ĐƯỢC bịa đặt thông tin.
            Phải từ chối trả lời.
            </think>
            
            Xin lỗi, tôi không tìm thấy thông tin về vấn đề này trong cơ sở dữ liệu y tế. Vui lòng liên hệ bác sĩ để được tư vấn chi tiết.
            
            📌 VÍ DỤ KHI TÀI LIỆU KHÔNG TRẢ LỜI ĐƯỢC CÂU HỎI:
            Câu hỏi: "Trẻ em bao nhiêu tuổi thì bị cấm uống rượu?"
            
            <think>
            Tìm thấy 2 tài liệu về tác hại rượu bia, nhưng KHÔNG có thông tin về độ tuổi cấm.
            Tài liệu không trả lời được câu hỏi cụ thể.
            KHÔNG được thêm kiến thức chung như "dưới 18 tuổi".
            Phải từ chối.
            </think>
            
            Xin lỗi, tôi không tìm thấy thông tin về độ tuổi cấm uống rượu bia trong cơ sở dữ liệu y tế. Vui lòng liên hệ bác sĩ để được tư vấn chi tiết.
            
            📌 VÍ DỤ KHI CÓ TÀI LIỆU TRẢ LỜI ĐƯỢC:
            Câu hỏi: "Tôi bị đau dạ dày 3 ngày, có nguy hiểm không?"
            
            <think>
            Tìm thấy 2 tài liệu:
            1. "Bệnh viêm loét dạ dày - Triệu chứng": Có đề cập "đau kéo dài >3 ngày là dấu hiệu cần khám"
            2. "Đau dạ dày - Phòng ngừa": Chỉ nói về chế độ ăn, không nói về mức độ nguy hiểm
            
            Phân tích:
            - Bệnh nhân đã đau 3 ngày → đúng ngưỡng cảnh báo theo tài liệu 1
            - Chưa biết thêm triệu chứng kèm theo (nôn, máu...) → cần hỏi thêm
            - Tài liệu 2 không giúp trả lời câu hỏi này → bỏ qua
            
            Kết luận: Cần khuyên bệnh nhân đi khám do đã đau 3 ngày (theo tài liệu 1)
            </think>
            
            Theo tài liệu về bệnh viêm loét dạ dày, đau dạ dày kéo dài trên 3 ngày là dấu hiệu cần thăm khám bác sĩ để loại trừ các nguy cơ nghiêm trọng. Anh/chị vui lòng cho biết thêm: có nôn, tiêu phân ra máu hoặc phân đen không? Nếu có, cần đi khám ngay.
            
            ✅ HÃY LUÔN TUÂN THỦ CẤU TRÚC <think>...</think> + Câu trả lời này.
        """).strip()

        human_template = """
        📚 LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ:
        {history}

        📖 TÀI LIỆU Y KHOA TÌM ĐƯỢC (BẮT BUỘC PHẢI DÙNG):
        {context}

        ❓ CÂU HỎI CỦA BỆNH NHÂN:
        {question}

        ⚠️ NHẮC NHỞ: Hãy mở thẻ <think>, CHẤM ĐIỂM ĐỘ LIÊN QUAN (0-3) cho từng tài liệu, bỏ qua tài liệu <2 điểm, đóng thẻ </think>, rồi trả lời bệnh nhân.
        TRẢ LỜI:"""

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(sys_instruction),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

        llm = ChatOllama(model=self.OLLAMA_MODEL, temperature=0)

        self.chain = (
            {
                "context": RunnableLambda(self._retrieve_and_store),
                "history": RunnableLambda(lambda x: self._format_history()),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        print("✅ Logic Engine đã sẵn sàng!")

    def _retrieve_and_store(self, query):
        """Retrieve context and force strict adherence if no documents found."""
        context = self.rag_engine.search(query, k=5)
        self.last_retrieved_context = context
        
        # If fallback (no documents found), explicitly forbid hallucination
        if "Không tìm thấy thông tin y tế phù hợp" in context:
            context = (
                "CRITICAL SYSTEM ALERT\n"
                "NO MEDICAL DOCUMENTS FOUND IN DATABASE FOR THIS QUERY.\n"
                "CONTEXT IS COMPLETELY EMPTY - NO INFORMATION AVAILABLE.\n\n"
                "⛔ ABSOLUTE PROHIBITION:\n"
                "- DO NOT use any external knowledge\n"
                "- DO NOT invent information\n"
                "- DO NOT pretend documents exist\n"
                "- DO NOT provide medical advice\n\n"
                "✅ REQUIRED RESPONSE FORMAT:\n"
                "<think>\n"
                "Không tìm thấy tài liệu nào trong cơ sở dữ liệu.\n"
                "Context hoàn toàn trống.\n"
                "KHÔNG ĐƯỢC sử dụng kiến thức bên ngoài.\n"
                "Phải từ chối trả lời.\n"
                "</think>\n\n"
                "Xin lỗi, tôi không tìm thấy thông tin về vấn đề này trong cơ sở dữ liệu y tế. "
                "Vui lòng liên hệ bác sĩ để được tư vấn chi tiết."
            )
        
        return context

    def _format_history(self):
        """Chuyển đổi list tin nhắn thành chuỗi văn bản"""
        formatted_text = ""
        recent_msgs = self.chat_history[-self.MAX_HISTORY:]
        for msg in recent_msgs:
            role = "Bệnh nhân" if isinstance(msg, HumanMessage) else "Bác sĩ"
            # Loại bỏ thẻ think khỏi lịch sử để tránh làm rối model lượt sau
            clean_content = re.sub(r'<think>.*?</think>', '', msg.content, flags=re.DOTALL).strip()
            formatted_text += f"- {role}: {clean_content}\n"
        return formatted_text if formatted_text else "Chưa có lịch sử."

    def chat_stream(self, query):
        if not self.chain:
            yield "⚠️ Lỗi: Hệ thống chưa khởi động."
            return

        full_response = ""
        start_time = time.time()
        
        try:
            # 1. Stream ra màn hình
            for chunk in self.chain.stream(query):
                full_response += chunk
                yield chunk
            
            # 2. Xử lý Log sau khi trả lời xong
            elapsed_time = time.time() - start_time
            
            # Tách Reasoning ra khỏi Answer bằng Regex (case-insensitive, allow spaces)
            # Support both XML tags <think>...</think> AND "Think:" format
            # Relaxed regex to allow unclosed tags at the end of string
            reasoning_match = re.search(r'<\s*think\s*>(.*?)(?:</\s*think\s*>|$)', full_response, re.DOTALL | re.IGNORECASE)
            
            # Additional check for "Think:" style if no tags found
            if not reasoning_match:
                # Look for "Think:" or "Thinking:" at start of string
                alt_match = re.match(r'^(?:Think|Thinking):\s*(.*?)(?:\n\n|\Z)', full_response, re.DOTALL | re.IGNORECASE)
                if alt_match:
                    reasoning_content = alt_match.group(1).strip()
                    # Remove the "Think: ..." part from final answer
                    final_answer = full_response[alt_match.end():].strip()
                    has_think = True
                else:
                    has_think = False
            else:
                 reasoning_content = reasoning_match.group(1).strip()
                 # Remove <think> content from answer. If unclosed, answer might be empty or partial.
                 final_answer = re.sub(r'<\s*think\s*>.*?(?:</\s*think\s*>|$)', '', full_response, flags=re.DOTALL | re.IGNORECASE).strip()
                 has_think = True

            if has_think:
                # ✅ Validate: Kiểm tra xem <think> có thực sự phân tích tài liệu không
                context_keywords = ["tài liệu", "nguồn", "theo", "bài viết", "mục"]
                has_analysis = any(kw in reasoning_content.lower() for kw in context_keywords)
                
                if not has_analysis:
                    warning = "\n\n⚠️ LƯU Ý: Model chưa phân tích kỹ tài liệu trong <think>"
                    yield warning
                    full_response += warning
                
                if len(final_answer.strip()) < 30:
                    warning = "\n\n⚠️ LƯU Ý: Câu trả lời quá ngắn, model có thể bỏ qua thông tin"
                    yield warning
                    full_response += warning
            else:
                reasoning_content = "❌ Không tìm thấy thẻ <think> hoặc từ khóa 'Think:' - Model KHÔNG tuân thủ quy tắc"
                # Remove any think-like tags from final answer for cleaner log
                final_answer = re.sub(r'<\s*/?\s*think\s*>', '', full_response, flags=re.IGNORECASE).strip()
                if not final_answer:
                    final_answer = full_response


            # Snapshot dữ liệu lúc gửi đi
            history_snapshot = self._format_history() 

            # Lưu vào log
            log_entry = {
                "turn": len(self.logs) + 1,
                "question": query,
                "context": self.last_retrieved_context,
                "history_snapshot": history_snapshot,
                "full_response": full_response,
                "reasoning": reasoning_content,
                "answer": final_answer,
                "time": f"{elapsed_time:.2f}s"
            }
            self.logs.append(log_entry)

            # Lưu vào bộ nhớ chat (Memory)
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=full_response))
            
        except Exception as e:
            # In lỗi ra console để dễ debug hơn
            print(f"❌ Error in chat_stream: {e}") 
            yield f"\n❌ Lỗi hệ thống: {str(e)}"

    def get_logs(self):
        return self.logs
    
    def get_last_sources(self):
        return self.last_retrieved_context
    
    def clear_history(self):
        self.chat_history = []
        self.last_retrieved_context = ""
        self.logs = []