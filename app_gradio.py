import gradio as gr
import threading
import cv2
import numpy as np
import json
import joblib
import os
import pandas as pd

# MediaPipe 0.10+ wrapper
from mediapipe_wrapper import FaceMesh

# --- IMPORTS FOR MEDICAL BOT ---
from bot_engine import MedicalLLM

# --- IMPORTS FOR BEAUTY ADVISOR ---
from measurements import FaceEvaluator 

# ==============================================================================
# 1. INITIALIZATION: MEDICAL CHATBOT
# ==============================================================================
print("⏳ Initializing Medical Bot...")
bot = MedicalLLM() 

# Initialize Recommender (share bot's embedding model to save RAM)
try:
    from recommender import Recommender
    # Pass the bot's embedding model to avoid loading it twice
    rec = Recommender("NER_trained_model", embedding_model=bot.embedding_model)
    # Global dictionary to hold results
    global_recs = {'articles': [], 'products': []} 
except Exception as e:
    print(f"⚠️ Recommender Init Failed: {e}")
    rec = None
    global_recs = {'articles': [], 'products': []}

print("✅ Medical Bot Ready!")

# ==============================================================================
# 2. INITIALIZATION: BEAUTY ADVISOR (ENHANCED)
# ==============================================================================
print("⏳ Initializing Beauty Engine...")

BEAUTY_DATA = {
    "AM": {"standards": None, "model": None, "order": []},
    "AF": {"standards": None, "model": None, "order": []}
}

for gender in ["AM", "AF"]:
    json_path = f'beauty_standards_{gender}_final.json'
    pkl_path = f'beauty_model_{gender}_best.pkl'
    
    # Load Standards
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            BEAUTY_DATA[gender]["standards"] = json.load(f)
            BEAUTY_DATA[gender]["order"] = list(BEAUTY_DATA[gender]["standards"].keys())
    
    # Load Model
    if os.path.exists(pkl_path):
        try:
            BEAUTY_DATA[gender]["model"] = joblib.load(pkl_path)
        except:
            print(f"⚠️ Could not load model for {gender}")

evaluator = FaceEvaluator()
face_mesh = FaceMesh(
    static_image_mode=True,
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5
)

def get_used_indices():
    indices = set()
    
    # 1. From LANDMARK_INDICES
    def collect(struct):
        if isinstance(struct, int): 
            indices.add(struct)
        elif isinstance(struct, (list, tuple)): 
            for x in struct: collect(x)
        elif isinstance(struct, dict): 
            for v in struct.values(): collect(v)
            
    collect(evaluator.idx)
    
    # 2. From ROI_DEFINITIONS
    collect(evaluator.rois)
    
    return indices

USED_LANDMARKS = get_used_indices()

print("✅ Beauty Engine Ready!")

# --- FULL DATABASE FROM APP.PY ---
FEATURE_NAMES_VN = {
    "R1_Mid_Lower_Ratio": "Tỷ lệ Mặt Giữa/Dưới",
    "R2_Cheek_Jaw_Ratio": "Tỷ lệ Má/Hàm",
    "R3_Face_Compactness": "Độ Gọn Mặt",
    "E4_Canthal_Tilt": "Góc Mắt (Tilt)",
    "E5_Eye_Aspect_Ratio": "Độ Mở Của Mắt",
    "E6_Interocular_Ratio": "Khoảng Cách 2 Mắt",
    "B7_Eyebrow_Height": "Độ Cao Chân Mày",
    "B8_Eyebrow_Arch": "Độ Cong Chân Mày",
    "B9_Eyebrow_Thickness": "Độ Dày Chân Mày",
    "N10_Nose_Width_Ratio": "Độ Rộng Cánh Mũi",
    "N11_Nose_Length_Ratio": "Độ Dài Mũi",
    "N12_Nose_Bridge_Ratio": "Độ Rộng Sống Mũi",
    "L13_Lip_Fullness": "Độ Dày Môi",
    "L14_Upper_Lower_Ratio": "Tỷ lệ Môi Trên/Dưới",
    "L15_Mouth_Nose_Ratio": "Tỷ lệ Miệng/Mũi",
    "J16_Chin_Compactness": "Độ Gọn Cằm",
    "S17_Skin_Homogeneity": "Độ Đều Màu Da",
    "S18_Skin_Lightness": "Độ Sáng Da",
    "S19_Skin_Contrast": "Độ Tương Phản Môi/Da",
    "A20_Philtrum_Ratio": "Tỷ lệ Nhân Trung",
    "A21_Nasolabial_Fold": "Rãnh Cười (Lão hóa)",
    "A22_Crows_Feet": "Vết Chân Chim",
    "A23_Periorbital_Aging": "Quầng Thâm/Bọng Mắt",
    "SYM_Symmetry_Index": "Độ Bất Đối Xứng"
}

ADVICE_DB = {
    "R1_Mid_Lower_Ratio": "Thấp: Mặt dưới dài (cân nhắc tóc mái). Cao: Mặt dưới ngắn (rẽ ngôi).",
    "R2_Cheek_Jaw_Ratio": "Thấp: Hàm bạnh (masseter botox/massage). Cao: Mặt V-line.",
    "R3_Face_Compactness": "Cao: Mặt dài (tạo phồng 2 bên tóc). Thấp: Mặt ngắn/bè (contour viền hàm).",
    "E4_Canthal_Tilt": "Âm: Mắt buồn (vẽ eyeliner xếch lên). Dương: Mắt xếch (kẻ mắt ngang).",
    "E5_Eye_Aspect_Ratio": "Thấp: Mắt hẹp (kích mí/lens giãn tròng). Cao: Mắt to tròn.",
    "E6_Interocular_Ratio": "Thấp: Mắt gần (highlight đầu mắt). Cao: Mắt xa (nhấn hốc mắt).",
    "B7_Eyebrow_Height": "Thấp: Mắt tối (tỉa gọn mặt dưới). Cao: Mặt thoáng (kẻ ngang).",
    "B8_Eyebrow_Arch": "Thấp: Lông mày ngang (hiền hòa). Cao: Lông mày xếch (sắc sảo).",
    "B9_Eyebrow_Thickness": "Thấp: Lông mày mỏng (điêu khắc/kẻ đậm). Cao: Rậm (tỉa gọn).",
    "N10_Nose_Width_Ratio": "Cao: Cánh mũi bè (contour sống mũi). Thấp: Mũi thon.",
    "N11_Nose_Length_Ratio": "Cao: Mũi dài (trưởng thành). Thấp: Mũi ngắn (trẻ trung).",
    "N12_Nose_Bridge_Ratio": "Cao: Sống mũi to. Thấp: Sống mũi mảnh.",
    "L13_Lip_Fullness": "Thấp: Môi mỏng (tràn viền/son bóng). Cao: Môi dày.",
    "L14_Upper_Lower_Ratio": "Lý tưởng ~0.6 (Môi dưới dày hơn môi trên).",
    "L15_Mouth_Nose_Ratio": "Thấp: Miệng nhỏ (son tràn viền). Cao: Miệng rộng (sang trọng).",
    "J16_Chin_Compactness": "Thấp: Cằm dài. Cao: Cằm ngắn/lẹm (filler).",
    "S17_Skin_Homogeneity": "Cao: Da không đều màu (Vitamin C/Retinol). Thấp: Da mịn.",
    "S18_Skin_Lightness": "(Chỉ số tham khảo tone da).",
    "S19_Skin_Contrast": "Thấp: Nhợt nhạt (dùng son màu đậm). Cao: Tươi tắn.",
    "A20_Philtrum_Ratio": "Cao: Nhân trung dài (dấu hiệu lão hóa/hack môi trên dày hơn).",
    "A21_Nasolabial_Fold": "Cao: Rãnh sâu (massage nâng cơ/filler).",
    "A22_Crows_Feet": "Cao: Nếp nhăn đuôi mắt (Retinol/Botox).",
    "A23_Periorbital_Aging": "Cao: Mắt thâm/bọng (ngủ đủ/kem mắt caffeine).",
    "SYM_Symmetry_Index": "Cao: Mặt lệch (nhai đều 2 bên, chỉnh nha, thay đổi tư thế ngủ)."
}

# ==============================================================================
# 3. LOGIC FUNCTIONS
# ==============================================================================

# --- MEDICAL BOT LOGIC ---
def chat_wrapper(message, history):
    if rec:
        def bg(): 
            global global_recs
            result = rec.recommend(message, 3) 
            global_recs = result
            
        threading.Thread(target=bg, daemon=True).start()
    
    partial_response = ""
    new_history = history.copy() if history else []

    for token in bot.chat_stream(message):
        partial_response += token
        current_exchange = new_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": partial_response}
        ]
        yield current_exchange

def update_inspector():
    logs = bot.get_logs()
    if not logs:
        return "No data", "No data", "No data", "No data"
    
    last_log = logs[-1]
    reasoning_view = f"⏱️ Thời gian: {last_log['time']}\n\n=== 🧠 SUY LUẬN ===\n{last_log['reasoning']}\n\n=== 🗣️ TRẢ LỜI ===\n{last_log['answer']}"
    return reasoning_view, last_log['context'], f"{last_log['history_snapshot']}\n\n=== QUESTION ===\n{last_log['question']}", logs

def get_recs():
    """Formats BOTH Articles and Products for the UI."""
    # 1. Format Articles
    articles = global_recs.get('articles', [])
    art_text = ""
    if not articles:
        art_text = "Chưa có gợi ý bài viết phù hợp."
    else:
        for r in articles:
            title = r.get('title', 'Không có tiêu đề')
            url = r.get('url') or r.get('link') or r.get('source')
            desc = r.get('desc') or r.get('section', '')
            score = r.get('score')

            art_text += f"📄 {title}\n"
            if score is not None:
                art_text += f"⭐ Điểm: {score}\n"
            art_text += f"🔗 {url}\n"
            if desc:
                art_text += f"ℹ️ {desc}\n"
            art_text += "\n"

    # 2. Format Products
    products = global_recs.get('products', [])
    prod_text = ""
    if not products:
        prod_text = "Chưa có gợi ý sản phẩm phù hợp."
    else:
        for p in products:
            name = p.get('name', 'Sản phẩm')
            rating = p.get('rating')
            desc = p.get('desc', '')
            url = p.get('url', '#')
            image = p.get('image')

            prod_text += f"🛍️ {name}\n"
            if rating is not None:
                prod_text += f"⭐ {rating}"
                if desc:
                    prod_text += f" | {desc}"
                prod_text += "\n"
            elif desc:
                prod_text += f"ℹ️ {desc}\n"
            if image:
                if not image.startswith("http") and not image.startswith("/"):
                     abs_path = os.path.abspath(os.path.join(os.getcwd(), image))
                     prod_text += f"![{name}]({abs_path})\n"
                elif image.startswith("/workspace/processed_dataset/product_images/"):
                     filename = os.path.basename(image)
                     abs_path = f"/workspace/product_images/{filename}"
                     prod_text += f"![{name}]({abs_path})\n"
                else:
                     prod_text += f"![{name}]({image})\n"
            
            prod_text += f"🔗 {url}\n\n"
            
    return art_text, prod_text

def clear_memory():
    global global_recs
    bot.clear_history()
    global_recs = {'articles': [], 'products': []}
    return [], "Cleared", "Cleared", "Cleared", [], ""

# [NEW HELPER] Filters advice based on status (High/Low)
def parse_advice(full_text, evaluation):
    if not full_text or evaluation == "Lý tưởng": return ""
    
    import re
    pattern = r"(thấp:|cao:|âm:|dương:)(.*?)(?=(thấp:|cao:|âm:|dương:|$))"
    matches = re.findall(pattern, full_text, re.IGNORECASE | re.DOTALL)
    
    target = evaluation.lower().replace(":", "")
    if target == "âm": target = "thấp"
    if target == "dương": target = "cao"

    for tag, content, _ in matches:
        clean_tag = tag.strip().lower().replace(":", "")
        if clean_tag == "âm": clean_tag = "thấp"
        if clean_tag == "dương": clean_tag = "cao"
        
        if clean_tag == target:
            return content.strip()
            
    return full_text 

# [NEW HELPER] Draws colorful groups and regions
def draw_beauty_overlays(image, lms, w, h, evaluator):
    overlay = image.copy()
    
    # 1. Draw Regions (Polygons)
    roi_colors = {
        'cheek_left': (200, 200, 255), 'cheek_right': (200, 200, 255),
        'lips_outer': (200, 200, 255),
        'nasolabial_left': (255, 200, 200), 'nasolabial_right': (255, 200, 200),
        'eye_bags_left': (200, 255, 200), 'eye_bags_right': (200, 255, 200),
        'crows_feet_left': (100, 255, 255), 'crows_feet_right': (100, 255, 255)
    }
    
    for name, indices in evaluator.rois.items():
        if name in roi_colors:
            pts = np.array([[int(lms[i].x * w), int(lms[i].y * h)] for i in indices], np.int32)
            cv2.fillPoly(overlay, [pts], roi_colors[name])
            cv2.polylines(image, [pts], True, [c-50 for c in roi_colors[name]], 1, cv2.LINE_AA)
            
    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

    # 2. Draw Dots (Groups)
    groups = {
        'Eyes': (list(evaluator.idx['left_eye'].values()) + list(evaluator.idx['right_eye'].values()), (0, 255, 255)),
        'Brows': (list(evaluator.idx['left_brow'].values()) + list(evaluator.idx['right_brow'].values()), (0, 165, 255)),
        'Nose': ([evaluator.idx['nose_tip'], evaluator.idx['nose_root']], (255, 0, 0)),
        'Lips': ([evaluator.idx['lip_top'], evaluator.idx['lip_bot']], (0, 0, 255)),
        'Jaw': ([evaluator.idx['chin_bottom'], evaluator.idx['jaw_left'], evaluator.idx['jaw_right']], (0, 255, 0))
    }
    
    for _, (indices, color) in groups.items():
        for i in indices:
            x, y = int(lms[i].x * w), int(lms[i].y * h)
            cv2.circle(image, (x, y), 2, color, -1)
            
    return image

# --- BEAUTY ADVISOR LOGIC (STATIC ONLY) ---
def analyze_beauty(input_image, gender_label, show_score):
    if input_image is None:
        return None, "Please upload an image", pd.DataFrame(), ""

    # 1. Map label to Code
    gender_code = "AM" if "Nam" in gender_label else "AF"
    
    # 2. Retrieve correct data
    data = BEAUTY_DATA.get(gender_code)
    standards = data["standards"]
    model = data["model"]
    feature_order = data["order"]

    h, w = input_image.shape[:2]
    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR) 
    
    # Process with MediaPipe
    try:
        results = face_mesh.process(input_image)
    except Exception as e:
        print(f"❌ MediaPipe Error: {e}")
        return input_image, "Error", pd.DataFrame([["Status", "Error", str(e)]], columns=["Đặc điểm", "Chỉ số", "Đánh giá"]), f"Lỗi xử lý hình ảnh: {e}"

    output_image = input_image.copy()
    
    score_str = "***"
    report_data = []
    advice_log = ""

    if results.multi_face_landmarks:
        lms = results.multi_face_landmarks[0].landmark
        
        draw_img_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        draw_img_bgr = draw_beauty_overlays(draw_img_bgr, lms, w, h, evaluator)
        output_image = cv2.cvtColor(draw_img_bgr, cv2.COLOR_BGR2RGB)

        # Calculate
        features = evaluator.calculate_features(img_bgr, lms)

        if features and standards:
            # 3. Predict Score (Only if toggled ON)
            if show_score and model:
                try:
                    vec = [features.get(k, 0) for k in feature_order]
                    pred = model.predict([vec])[0]
                    score_str = f"{pred:.2f}"
                except: score_str = "Error"
            elif not show_score:
                score_str = "Điểm: ****"

            # 4. Generate Report & Advice
            for key in feature_order:
                val = features.get(key, 0)
                std_dat = standards.get(key, {})
                ideal = std_dat.get('ideal', 0)
                
                status = "Lý tưởng"
                is_issue = False
                
                if abs(val - ideal) > (ideal * 0.15):
                    if val > ideal: status = "Cao"
                    else: status = "Thấp"
                    is_issue = True
                
                vn_name = FEATURE_NAMES_VN.get(key, key)
                report_data.append([vn_name, f"{val:.2f}", status])

                if is_issue:
                    full_adv = ADVICE_DB.get(key, "")
                    specific_adv = parse_advice(full_adv, status)
                    if specific_adv:
                        advice_log += f"📌 {vn_name} ({status}):\n   {specific_adv}\n\n"
        
        if not advice_log: advice_log = "✅ Gương mặt rất cân đối, không có khuyến nghị cụ thể!"

    else:
        advice_log = "⚠️ Không tìm thấy khuôn mặt. Vui lòng thử ảnh khác."

    df = pd.DataFrame(report_data, columns=["Đặc điểm", "Chỉ số", "Đánh giá"])
    return output_image, score_str, df, advice_log

# ==============================================================================
# 4. GRADIO UI LAYOUT
# ==============================================================================
custom_css = """
#chatbot {height: 550px !important; overflow-y: auto; border: 1px solid #e5e7eb;}
#inspector_text {font-family: monospace; font-size: 12px;}
#score_box {font-size: 32px; font-weight: bold; color: #EAB308; text-align: center;}
"""

with gr.Blocks(css=custom_css, title="AI Integrated System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🏥 & ✨ Hệ Thống AI Tổng Hợp (Medical + Beauty)")
    
    with gr.Tabs():
        # --- TAB 1: MEDICAL CHATBOT ---
        with gr.TabItem("🏥 Bác Sĩ AI (Chatbot)"):
            with gr.Row():
                with gr.Column(scale=5):
                    chatbot = gr.Chatbot(elem_id="chatbot", label="Hội Thoại")
                    msg = gr.Textbox(placeholder="Nhập câu hỏi bệnh lý...", show_label=False)
                    with gr.Row():
                        submit_btn = gr.Button("Gửi tin nhắn", variant="primary")
                        clear_btn = gr.Button("🗑️ Xóa hội thoại")
                with gr.Column(scale=4):
                    gr.Markdown("### 🔍 Real-time Inspector")
                    with gr.Tabs():
                        with gr.TabItem("📄 Articles"):
                            rec_box = gr.Textbox(label="Recommendations", lines=15, elem_id="inspector_text")
                        with gr.TabItem("🛒 Products"):
                            prod_box = gr.Markdown(label="Gợi ý Sản phẩm", show_label=True)
                        with gr.TabItem("🧠 Reasoning"):
                            reasoning_box = gr.Textbox(label="Thinking Process", lines=20, elem_id="inspector_text")
                        with gr.TabItem("📚 Context"):
                            context_box = gr.Textbox(label="Retrieved Docs", lines=20, elem_id="inspector_text")
                        with gr.TabItem("⚙️ Stats"):
                            stats_box = gr.Textbox(label="Snapshot", lines=20, elem_id="inspector_text")
                        with gr.TabItem("📊 Logs"):
                            log_json = gr.JSON(label="Full Logs")
            
            # Events
            msg_submit = msg.submit(chat_wrapper, [msg, chatbot], [chatbot])
            btn_submit = submit_btn.click(chat_wrapper, [msg, chatbot], [chatbot])
            
            # Update Inspector Tabs
            msg_submit.then(update_inspector, None, [reasoning_box, context_box, stats_box, log_json])
            btn_submit.then(update_inspector, None, [reasoning_box, context_box, stats_box, log_json])
            
            # Recommendations
            msg_submit.then(get_recs, None, [rec_box, prod_box])
            btn_submit.then(get_recs, None, [rec_box, prod_box])
            clear_btn.click(clear_memory, None, [chatbot, reasoning_box, context_box, stats_box, log_json, rec_box])

        # --- TAB 2: BEAUTY ADVISOR (NO CAMERA) ---
        with gr.TabItem("✨ AI Beauty Advisor (Vision)"):
            gr.Markdown("### Phân tích khuôn mặt chuẩn Asian Beauty")
            with gr.Row():
                # Left: Input
                with gr.Column(scale=1):
                    # Removed "webcam" from sources
                    beauty_input = gr.Image(sources=["upload"], label="Upload Image", type="numpy")
                    
                    with gr.Row():
                        gender_radio = gr.Radio(["Nam (AM)", "Nữ (AF)"], label="Giới tính", value="Nam (AM)")
                        show_score_chk = gr.Checkbox(label="Hiện Điểm AI", value=False)
                        # Removed stream checkbox
                    
                    beauty_btn = gr.Button("🔍 Phân Tích Ngay", variant="primary")

                # Right: Output
                with gr.Column(scale=1):
                    score_out = gr.Textbox(label="Điểm AI Chấm", elem_id="score_box")
                    beauty_output = gr.Image(label="Landmarks")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📊 Chi tiết chỉ số")
                    result_table = gr.Dataframe(headers=["Đặc điểm", "Chỉ số", "Đánh giá"], wrap=True)
                
                with gr.Column():
                    gr.Markdown("### 💡 Lời khuyên chuyên sâu (Tự động tổng hợp)")
                    advice_out = gr.Textbox(label="Recommendation", lines=15)

            # Beauty Events (Static Only)
            beauty_btn.click(
                analyze_beauty, 
                inputs=[beauty_input, gender_radio, show_score_chk], 
                outputs=[beauty_output, score_out, result_table, advice_out]
            )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", share=True, allowed_paths=["/workspace/product_images"])