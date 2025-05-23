# from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# os.environ["GROQ_API_KEY"] = "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"

# OpenAI API 키 설정
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
# llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)

# 여행 추천 함수
def generate_trip_plan(trip_input, congestion_texts: str) -> str:
    only_type_0 = ""
    trip = ""
    answer_ex = ""
    only_type_1 = ""
    only_type_2 = ""
    congestion_ignore = f"[혼잡한 지역 정보 (반드시 피할 것)]\n{congestion_texts}"
    # 당장여행
    if trip_input.question_type == 0:
        trip = f"여행 시작 위치: {trip_input.start_place}"
        answer_ex = """
    🕙 00:00 - 관광지 (관광지 간단 설명)
       - 🚶 도보 이동 (00분)

    🕚 00:00 - 관광지 (관광지 간단 설명)
       - 🚇 지하철 이동 (00분, 약 000원)
       - 🚕 택시 이동 (00분, 약 000원)

    🕛 00:30 - 관광지 (관광지 간단 설명)
       - 🍽️ 점심 포함
       - 🚶 도보 이동 (00분)

    🕑 00:00 - 관광지 (관광지 간단 설명)
       -  체험료 (약 000원)
       - 🚇 지하철 이동 (00분, 약 000원)
       - 🚕 택시 이동 (00분, 약 000원)
    🕓 00:00 - 관광지 (관광지 간단 설명)
       - 🚌 버스 이동 (00분, 약 000원)
       - 🚇 지하철 이동 (00분, 약 000원)
       - 🚕 택시 이동 (00분, 약 000원)

    🕕 00:00 - 여행 종료"""
    # 당일코스
    elif trip_input.question_type == 1:
        trip = f"여행 시작 위치: {trip_input.start_place}"
        only_type_1 = ""
        answer_ex = """
    🕙 00:00 - 관광지 (관광지 간단 설명)
       - 🚶 도보 이동 (00분)

    🕚 00:00 - 관광지 (관광지 간단 설명)
       - 🚇 지하철 이동 (00분, 약 000원)
       - 🚕 택시 이동 (00분, 약 000원)

    🕛 00:30 - 관광지 (관광지 간단 설명)
       - 🍽️ 점심 포함
       - 🚶 도보 이동 (00분)

    🕑 00:00 - 관광지 (관광지 간단 설명)
       -  체험료 (약 000원)
       - 🚇 지하철 이동 (00분, 약 000원)
       - 🚕 택시 이동 (00분, 약 000원)
    🕓 00:00 - 관광지 (관광지 간단 설명)
       - 🚌 버스 이동 (00분, 약 000원)
       - 🚇 지하철 이동 (00분, 약 000원)
       - 🚕 택시 이동 (00분, 약 000원)
    🕕 00:00 - 여행 종료"""
    # 숙박여행
    elif trip_input.question_type == 2:
        trip = f"숙소 위치: {trip_input.start_place}"
        only_type_2 = "여행 시작시 숙소 체크인(짐 맡기기)후 일정 시작, 여행 경로 설정시 숙소 위치 필수 고려, 마지막날 일정 시작 전 체크아웃 진행, 자유시간은 숙소인근의 번화가에서 당일 일정 마지막에 부여, 마지막 날 체크아웃 이후 짐을 고려해서 활동적인 관광지 및 힘든 이동 경로 제한"
        answer_ex = """
    🏨 숙소: 남산 아래 위치로 교통 편리, 명동과 을지로 접근성 우수

    📅 Day 1 - 도심 중심 관광
    🕓 00:00 - 숙소 체크인 

    🕙 00:00 - 관광지 (관광지 간단 설명)
       - 🚇 지하철 이동 (00분, 약 000원)
       - 🚕 택시 이동 (00분, 약 000원)

    🕛 00:00 - 관광지 (관광지 간단 설명)
       -  체험료 (약 000원)
       - 🚶 도보 이동 (00분)

    🕑 00:00 - 관광지 (관광지 간단 설명)
       - 🍽️ 식사 포함
       - 🚶 도보 이동 (00분)
       - 🚕 택시 이동 (00분, 약 000원)

    🕓 00:00 - 관광지 (관광지 간단 설명)
       - 🚌 버스 이동 (00분, 약 000원)
       - 🚕 택시 이동 (00분, 약 000원)
    🕕 00:00 - 숙소 복귀

    ---

    📅 Day 2 - 관광 테마 설명

    🕙 00:00 - 관광지 (관광지 간단 설명)
       - 🚇 지하철 이동 (00분, 약 000원)
       - 🚕 택시 이동 (00분, 약 000원)

    🕛 00:00 - 관광지 (관광지 간단 설명)
       - 🚶 도보 이동 (00분)
       - 🚕 택시 이동 (00분, 약 000원)

    🕑 00:00 - 관광지 (관광지 간단 설명)
       - 🚇 지하철 이동 (00분, 약 000원)
       - 🚕 택시 이동 (00분, 약 000원)

    🕓 00:00 - 관광지 (관광지 간단 설명)

    🕔 00:00~00:00 - 자유시간

    🕕 00:00 - 숙소 복귀

    ---

    📅 Day 3 - 관광 테마 설명

    🕓 11:00 - 짐 정리 및 체크아웃

    🕙 00:00 - 관광지 (관광지 간단 설명)
       - 🚶 도보 출발 (숙소 인근)

    🕛 00:00 - 관광지 (관광지 간단 설명)
       - 🎫 전망대 입장 (약 000원)

    🕑 00:00 - 관광지 (관광지 간단 설명)
       - 🚌 이동 (10분, 약 1,250원)
       - 🚕 택시 이동 (00분, 약 000원)"""

    prompt = f"""
    [사용자 여행 정보]
    - 출생년도: {trip_input.birth_year}
    - 성별: {trip_input.gender}
    - MBTI: {trip_input.mbti}
    - 여행 시간: {trip_input.start_time} ~ {trip_input.finish_time}
    - {trip}
    - 요청사항: {trip_input.optional_request}


    위 정보를 바탕으로 서울 여행 코스를 구성해주세요.


    ✅ 조건:
    - 각 일정마다 장소, 활동, 이동수단, 소요시간, 예상 비용 포함
    - 요청사항 반드시 반영
    - 각 장소별 추천 이유 간단히 명시
    - {only_type_0}{only_type_1}{only_type_2}

    {congestion_ignore}

    ✅ 출력 형식 예시:

    📌 Tip:
    {trip_input.mbti} 성향에 맞춰 일정 구성 및 추천 이유를 제공하세요.

    ⏰ 일정표 ({trip_input.start_time} 시작 ~ {trip_input.finish_time} 종료)

    {answer_ex}

    """

    return llm.invoke(prompt)

