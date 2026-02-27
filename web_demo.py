import streamlit as st
import uuid
import json
import numpy as np
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
# THƯ VIỆN MỚI ĐỂ TẠO MENU ICON CHUYÊN NGHIỆP
from streamlit_option_menu import option_menu

# ================== 1. CẤU HÌNH API & TRANG ==================
# ĐIỀN API KEY CỦA BẠN VÀO ĐÂY
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
client = Groq()

st.set_page_config(page_title="LigoAI - Trợ Lý Pháp Lý", layout="wide", page_icon="✨")

# ================== 2. SIÊU CẤP CSS (GEMINI DARK MODE CLONE) ==================
st.markdown("""
    <style>
    /* Import Font chữ hiện đại của Google */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    /* === TỔNG THỂ === */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        color: #E3E3E3 !important; /* Màu chữ trắng xám của Gemini */
    }
    /* Nền chính tối màu #131314 */
    .stApp {
        background-color: #131314;
    }

    /* === SIDEBAR (THANH BÊN) === */
    [data-testid="stSidebar"] {
        background-color: #1E1F20 !important; /* Màu xám tối đặc trưng */
        border-right: 1px solid #333538 !important;
        padding-top: 20px;
    }
    /* Ẩn nút đóng mở sidebar mặc định cho gọn */
    [data-testid="collapsedControl"] {display: none;}

    /* CSS cho menu option-menu (New chat, My stuff) */
    .nav-link {
        border-radius: 8px !important;
        margin-bottom: 5px !important;
        font-weight: 500 !important;
        color: #E3E3E3 !important;
    }
    .nav-link:hover {
        background-color: #333538 !important;
    }
    .nav-link-selected {
        background-color: #282A2C !important;
        color: #A8C7FA !important; /* Màu xanh sáng khi chọn */
    }

    /* CSS cho nút Lịch sử chat và Gợi ý */
    div[data-testid="stButton"] button {
        text-align: left;
        height: auto;
        white-space: normal;
        padding: 10px 14px;
        border-radius: 8px; /* Bo góc nhẹ */
        border: none;
        background-color: transparent; /* Nền trong suốt */
        color: #C4C7C5; /* Chữ màu xám nhạt */
        transition: all 0.1s ease-in-out;
        font-size: 14px;
        margin-bottom: 2px;
    }
    /* Hiệu ứng hover mượt mà */
    div[data-testid="stButton"] button:hover {
        background-color: #333538;
        color: #E3E3E3;
    }
    /* Nút gợi ý ở màn hình chính thì cho có viền nhẹ */
    .suggestion-btn div[data-testid="stButton"] button {
         border: 1px solid #444746;
         background-color: #1E1F20;
         padding: 15px;
         border-radius: 12px;
    }
    .suggestion-btn div[data-testid="stButton"] button:hover {
         border-color: #8AB4F8;
         background-color: #282A2C;
    }

    /* === PHẦN CHAT CHÍNH === */
    /* Tiêu đề chào mừng */
    .welcome-text {
        font-size: 3rem; font-weight: 600;
        background: linear-gradient(90deg, #8AB4F8, #A8C7FA);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .sub-welcome { font-size: 1.5rem; color: #8e918f; font-weight: 500;}

    /* Ô nhập liệu chat */
    .stChatInput textarea {
        background-color: #1E1F20 !important;
        border: 1px solid #444746 !important;
        color: #E3E3E3 !important;
        border-radius: 24px !important; /* Bo tròn hẳn như Gemini */
        padding: 12px 20px !important;
    }
    .stChatInput textarea:focus {
        border-color: #8AB4F8 !important;
        box-shadow: none !important;
    }

    /* Các khung hiển thị luật, rủi ro */
    div[data-testid="stContainer"] {border: none;}
    div[data-testid="stMetric"] {background-color: #1E1F20; border: 1px solid #333538; border-radius: 12px;}
    .law-quote {background-color: #282A2C; border-left: 3px solid #8AB4F8; padding: 15px; border-radius: 8px; font-style: italic; margin-top:10px; font-size: 14px;}

    /* Ẩn các thành phần thừa */
    #MainMenu, header, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


# ================== 3. LOAD DATABASE & VECTOR ==================
@st.cache_resource
def load_embedder(): return SentenceTransformer('keepitreal/vietnamese-sbert')


embedder = load_embedder()


@st.cache_data
def load_laws():
    try:
        with open("legal_data.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []


laws = load_laws()


@st.cache_resource
def load_embeddings(_laws):
    if not _laws: return None
    law_texts = [item.get("content", "") for item in _laws]
    return embedder.encode(law_texts)


law_embeddings = load_embeddings(laws)


def retrieve_law_vector(query, top_k=3):
    if law_embeddings is None: return []
    query_embedding = embedder.encode([query])[0]
    scores = [(np.dot(query_embedding, le) / (np.linalg.norm(query_embedding) * np.linalg.norm(le)), idx) for idx, le in
              enumerate(law_embeddings)]
    scores.sort(reverse=True)
    return [laws[idx] for score, idx in scores[:top_k]]


# ================== 4. SIDEBAR: GEMINI STYLE ==================
if "conversations" not in st.session_state: st.session_state.conversations = {}
if "current_chat" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.conversations[new_id] = []
    st.session_state.current_chat = new_id

with st.sidebar:
    # MENU ĐIỀU HƯỚNG CHÍNH VỚI ICON CHUYÊN NGHIỆP (KHÔNG DÙNG EMOJI)
    selected_nav = option_menu(
        menu_title=None,
        options=["New chat", "My stuff"],
        icons=["plus-circle", "collection"],  # Sử dụng Bootstrap Icons
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#8AB4F8", "font-size": "18px"},
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px", "--hover-color": "#333538"},
            "nav-link-selected": {"background-color": "#282A2C"},
        }
    )

    if selected_nav == "New chat":
        # Logic tạo chat mới (Chỉ chạy khi người dùng thực sự cần reset)
        if st.session_state.conversations[st.session_state.current_chat]:
            new_id = str(uuid.uuid4())
            st.session_state.conversations[new_id] = []
            st.session_state.current_chat = new_id
            st.rerun()

    st.markdown("---")  # Đường kẻ ngang mờ
    st.markdown("<p style='font-size: 14px; font-weight: 600; color: #E3E3E3; margin-bottom: 10px;'>Recents</p>",
                unsafe_allow_html=True)

    # DANH SÁCH LỊCH SỬ CHAT (Nút bấm text gọn gàng)
    chat_ids = list(st.session_state.conversations.keys())
    # Đảo ngược để hiện cái mới nhất lên đầu
    for chat_id in reversed(chat_ids):
        messages = st.session_state.conversations[chat_id]
        # Lấy 30 ký tự đầu của câu hỏi đầu tiên làm tiêu đề
        title = messages[0]["content"][:30] + "..." if messages else "Cuộc hội thoại mới"

        # Nút bấm chuyển đổi lịch sử
        if st.button(title, key=chat_id, use_container_width=True):
            st.session_state.current_chat = chat_id
            st.rerun()

    # Phần cài đặt mô phỏng
    st.markdown("---")
    with st.expander("⚙️ Settings & parameters"):
        biz_type = st.selectbox("Ngành nghề:", ["Bán lẻ, Tạp hóa", "F&B (Nhà hàng)", "Dịch vụ", "Sản xuất"])
        revenue_val = st.slider("Doanh thu (Triệu/năm):", 0, 2000, 150)

# ================== 5. GIAO DIỆN CHÍNH & CHAT ==================
current_messages = st.session_state.conversations[st.session_state.current_chat]
suggestion_clicked = None

# MÀN HÌNH CHÀO MỪNG (Khi chưa có tin nhắn)
if not current_messages:
    st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)  # Khoảng trống
    st.markdown('<p class="welcome-text">Xin chào, tôi là LigoAI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-welcome">Tôi có thể giúp gì cho công việc kinh doanh của bạn hôm nay?</p>',
                unsafe_allow_html=True)
    st.markdown('<div style="margin-bottom: 50px;"></div>', unsafe_allow_html=True)

    # Các nút gợi ý (Được bọc class để CSS làm đẹp riêng)
    st.markdown('<div class="suggestion-btn">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🛒 Mở tiệm tạp hóa doanh thu 150tr thì đóng thuế gì?", use_container_width=True):
            suggestion_clicked = "Mở tiệm tạp hóa doanh thu 150tr thì đóng thuế gì?"
        if st.button("📜 Thủ tục đăng ký hộ kinh doanh cần giấy tờ gì?", use_container_width=True):
            suggestion_clicked = "Thủ tục đăng ký hộ kinh doanh cần giấy tờ gì?"
    with c2:
        if st.button("💰 Doanh thu dưới 100 triệu có phải nộp thuế không?", use_container_width=True):
            suggestion_clicked = "Doanh thu dưới 100 triệu có phải nộp thuế không?"
        if st.button("⚠️ Mức phạt chậm nộp tờ khai thuế môn bài là bao nhiêu?", use_container_width=True):
            suggestion_clicked = "Mức phạt chậm nộp tờ khai thuế môn bài là bao nhiêu?"
    st.markdown('</div>', unsafe_allow_html=True)

# HIỂN THỊ LỊCH SỬ CHAT
for msg in current_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("retrieved"):
            st.write("")
            # Phần đánh giá rủi ro (Giữ nguyên logic cũ)
            with st.container(border=True):
                st.markdown("##### 📊 Đánh giá tuân thủ")
                c_risk, c_rev = st.columns([1, 2])
                with c_risk:
                    if any(w in msg["content"].lower() for w in ["phạt", "cưỡng chế"]):
                        st.error("⚠️ Rủi ro: CAO")
                    else:
                        st.success("✅ Rủi ro: THẤP")
                with c_rev:
                    st.info(f"Áp dụng mức doanh thu: {msg.get('revenue')} triệu/năm")

            # Phần trích dẫn luật (Giao diện mới)
            with st.expander("những căn cứ pháp lý liên quan"):
                for item in msg["retrieved"]:
                    t = item.get("title", "Văn bản")
                    c = item.get("content", "")
                    st.markdown(f"**{t}**")
                    st.markdown(f'<div class="law-quote">{c}</div>', unsafe_allow_html=True)

# XỬ LÝ INPUT VÀ AI
SYSTEM_PROMPT = """
Bạn là LigoAI - Chuyên gia tư vấn THUẾ cho Hộ kinh doanh.
Cấu trúc trả lời BẮT BUỘC (Dùng Markdown):
1. 🎯 NHẬN ĐỊNH NGHĨA VỤ THUẾ: Kết luận ngay dựa trên input.
2. 📖 CĂN CỨ & GIẢI THÍCH: Trích dẫn nguyên văn luật từ CONTEXT (trong ngoặc kép) rồi giải thích bình dân.
3. 🛠️ HƯỚNG DẪN THỦ TỤC: Liệt kê các bước làm hồ sơ, nơi nộp, hạn nộp.
4. 💡 CẢNH BÁO RỦI RO: Mức phạt cụ thể nếu vi phạm.
"""

user_input = st.chat_input("Nhập vấn đề pháp lý của bạn tại đây...")
prompt = user_input or suggestion_clicked

if prompt:
    current_messages.append({"role": "user", "content": prompt})
    st.rerun()  # Rerun để hiển thị câu hỏi của user ngay lập tức

# Logic gọi AI (Chạy sau khi rerun)
if current_messages and current_messages[-1]["role"] == "user":
    last_prompt = current_messages[-1]["content"]
    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        full_res = ""
        retrieved = retrieve_law_vector(last_prompt)

        if not retrieved:
            full_res = "Vấn đề này nằm ngoài phạm vi dữ liệu Thuế & Thủ tục Hộ kinh doanh của LigoAI."
            msg_placeholder.markdown(full_res)
        else:
            context = "\n".join([f"{i.get('title')}:\n{i.get('content')}" for i in retrieved])
            rag_prompt = f"Lĩnh vực {biz_type}, doanh thu {revenue_val} triệu.\nCONTEXT:\n{context}\nUSER QUERY:\n{last_prompt}"

            try:
                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": rag_prompt}],
                    stream=True, temperature=0.1
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_res += chunk.choices[0].delta.content
                        msg_placeholder.markdown(full_res + "▌")
                msg_placeholder.markdown(full_res)
            except Exception as e:
                full_res = f"⚠️ Lỗi kết nối: {e}"
                msg_placeholder.markdown(full_res)

        current_messages.append(
            {"role": "assistant", "content": full_res, "retrieved": retrieved, "revenue": revenue_val})
    st.rerun()