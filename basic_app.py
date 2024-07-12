import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 엑셀 파일 경로 설정 (실제 파일 경로로 변경하세요)
EXCEL_FILE_PATH = '수탐 엑셀.xlsx'  # 실제 파일 경로로 변경

# 엑셀 파일 로드 및 데이터 전처리
@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path)
    df['Q'] = df['Q'].fillna('')
    df['A'] = df['A'].fillna('')
    return df

df = load_data(EXCEL_FILE_PATH)

# TF-IDF 벡터화 도구 학습
@st.cache_resource
def train_vectorizer(data):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(data['Q'].tolist())
    return vectorizer, vectors

question_vectorizer, question_vector = train_vectorizer(df)

# 유사 질문 찾기 함수
def get_most_similar_question(user_question, threshold):
    new_sen_vector = question_vectorizer.transform([user_question])
    simil_score = cosine_similarity(new_sen_vector, question_vector)
    if simil_score.max() < threshold:
        return None, "유사한 질문을 찾을 수 없습니다."
    else:
        max_index = simil_score.argmax()
        most_similar_question = df['Q'].tolist()[max_index]
        most_similar_answer = df['A'].tolist()[max_index]
        return most_similar_question, most_similar_answer

# 세션 상태 초기화
if 'active_button' not in st.session_state:
    st.session_state.active_button = None

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]

st.title("나를 소개합니다")

# 사이드바에 버튼 배열
st.sidebar.header("버튼을 클릭해보세요")

large_buttons = {
    "나의 아바타": "Untitled video.mp4",  # 이 경로를 실제 아바타 비디오 파일로 변경하세요.
    "나를 표현한 음악": "개발자의 꿈.mp3"  # 이 경로를 실제 음악 파일로 변경하세요.
}

small_buttons = {
    "나의 장점": "집중력이 저의 장점입니다.",
    "희망 진로": "소프트웨어 엔지니어가 되고 싶습니다.",
    "좋아하는 것": "게임, 독서, 명상을 좋아합니다.",
    "싫어하는 것": "게으른 습관을 싫어합니다.",
    "자기 소개": "저는 실용적인 것들을 좋아합니다.",
    "진로 준비": "저의 진로를 위해 프로그래밍을 공부하고 있습니다.",
    "취미 활동": "저의 취미는 그림 그리기입니다.",
    "MBTI": "제 MBTI는 INTP입니다."
}

for button, content in large_buttons.items():
    if st.sidebar.button(button, key=button):
        st.session_state.active_button = button if st.session_state.active_button != button else None
    if st.session_state.active_button == button:
        if button == "나를 소개하는 아바타":
            st.sidebar.video(content)
        elif button == "나를 표현한 음악":
            st.sidebar.audio(content)

for button, content in small_buttons.items():
    if st.sidebar.button(button, key=button):
        st.session_state.active_button = button if st.session_state.active_button != button else None
    if st.session_state.active_button == button:
        st.markdown(f"<div class='section'>{content}</div>", unsafe_allow_html=True)

# 유사도 임계값 슬라이더 추가
threshold = st.slider("유사도 임계값", 0.0, 1.0, 0.43)

# 사용자 입력을 받는 입력 창
user_input = st.text_input("질문을 입력하세요:")

# 검색 버튼
if st.button("검색", key="search"):
    if user_input:
        # 유사 질문 찾기
        similar_question, answer = get_most_similar_question(user_input, threshold)
        
        if similar_question:
            st.session_state.conversation_history.append({"role": "assistant", "content": f"유사한 질문: {similar_question}"})
            st.session_state.conversation_history.append({"role": "assistant", "content": answer})
            st.write(f"**유사한 질문:** {similar_question}")
            st.write(f"**답변:** {answer}")
        else:
            st.write("유사한 질문을 찾을 수 없습니다.")

# 이전 대화 보기 버튼
if st.button("이전 대화 보기", key="view_history"):
    st.write("### 대화 기록")
    for msg in st.session_state.conversation_history:
        role = "You" if msg["role"] == "user" else "Assistant"
        st.write(f"**{role}:** {msg['content']}")

# 새 검색 시작 버튼
if st.button("새 검색 시작", key="new_search"):
    st.session_state.conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]
    st.experimental_rerun()
