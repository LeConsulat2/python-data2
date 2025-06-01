# Import necessary libraries
import streamlit as st
import pandas as pd
import json
import re
from dotenv import load_dotenv
import os
import google.generativeai as genai # 주석: Gemini API를 위한 새로운 임포트
import numpy as np # 주석: 코드 실행 시 numpy를 사용할 수 있도록 미리 임포트 (exec 환경에 주입)

# 주석: 환경 변수 로드 (예: .env 파일에서 GEMINI_API_KEY 로드)
load_dotenv()

# 주석: Gemini API 키 설정. 환경 변수에서 키가 없는 경우 앱을 중단합니다.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("`GEMINI_API_KEY` environment variable not set. Please set it in your `.env` file.")
    st.stop() # Stops the Streamlit app if API key is missing

# 주석: Gemini 모델 초기화
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}. Please check your API key.")
    st.stop()

# --- LLM API Configuration ---
# 주석: LLM 모델 이름 정의. Gemini 1.5 Flash는 빠른 응답과 대규모 컨텍스트를 제공합니다.
GEMINI_MODEL_NAME = "gemini-1.5-flash"
TEMPERATURE = 0.2  # 주석: LLM의 창의성 조절. 0에 가까울수록 일관되고 예측 가능한 답변.
MAX_OUTPUT_TOKENS = 4096 # 주석: LLM 응답의 최대 토큰 수. 코드 생성에 충분한 크기.

# --- LLM Call Function ---
def call_gemini_api(prompt: str) -> str:
    """
    # 주석: 주어진 프롬프트로 Gemini LLM을 동기적으로 호출합니다.
    # 주석: 이는 사용자 메시지를 LLM에게 전달하는 일반적인 헬퍼 함수입니다.
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        # 주석: LLM 응답 생성을 위한 구성 설정
        generation_config = {
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "response_mime_type": "text/plain", # 주석: 일반 텍스트 응답을 기대합니다.
        }
        
        # 주석: LLM에 보낼 메시지 (단일 사용자 메시지)
        messages = [{"role": "user", "parts": [prompt]}]
        
        st.info(f"Calling Gemini {GEMINI_MODEL_NAME}...") # 주석: LLM 호출 시작을 사용자에게 알립니다.
        
        response = model.generate_content(
            messages,
            generation_config=generation_config
        )
        
        # 주석: 응답 텍스트를 추출하여 반환
        return response.text
    except Exception as e:
        # 주석: LLM API 호출 중 발생한 오류를 처리합니다.
        st.error(f"Error calling Gemini API: {e}. Please check your API key or network connection.")
        return f"ERROR: LLM API call failed: {e}"

# --- Prompt Engineering Functions ---

def generate_code_prompt(user_query: str, df_preview: dict, df_types: dict) -> str:
    """
    # 주석: 사용자 질의와 DataFrame의 미리보기 및 타입 정보를 기반으로 Python 코드 생성을 위한 프롬프트를 생성합니다.
    # 주석: 이 프롬프트는 LLM이 견고하고 정확한 Pandas 코드를 생성하도록 상세한 지침을 포함합니다.
    """
    # 주석: DataFrame 미리보기 및 타입 정보를 JSON 문자열로 변환합니다.
    # 주석: ensure_ascii=False는 한국어 문자를 올바르게 포함하기 위해 중요합니다.
    preview_str = json.dumps(df_preview, ensure_ascii=False, indent=2)
    types_str = json.dumps(df_types, ensure_ascii=False, indent=2)

    # 주석: 코드 생성 프롬프트 구성. LLM의 역할을 명확히 하고, 기대하는 코드의 특성을 상세히 설명합니다.
    prompt = f"""
    You are an expert Python data analyst assistant. Your task is to generate robust and correct Pandas Python code to analyze a DataFrame named `df`.
    The code will be executed in an environment where `pandas` is imported as `pd`, and `numpy` is imported as `np`.

    Here's a preview of the DataFrame `df` (first 5 rows):
    ```json
    {preview_str}
    ```

    And here are the detected data types for each column in `df`:
    ```json
    {types_str}
    ```

    The user's specific request for analysis is: "{user_query}"

    **Crucial Instructions for Code Generation:**
    1.  **Assume `df` is loaded:** Your code should operate on a DataFrame variable already named `df`.
    2.  **Return `final_df`:** The ultimate result of your analysis MUST be assigned to a new Pandas DataFrame variable named `final_df`. This `final_df` will be used for further summarization.
    3.  **Contextual Results:** Even if the user's query asks for a single metric (e.g., maximum, minimum, top 1 item), the `final_df` must provide **full relevant context**. For example, if the query is "Which district has the highest building floor count?", `final_df` should return a DataFrame that includes the district, the floor count, and possibly other related columns, sorted or filtered to highlight the answer, not just the name of the top district. This provides a richer and verifiable context.
    4.  **Handle Data Issues Robustly:**
        *   **Missing Values (NaNs):** Be prepared to handle `NaN`s using methods like `.dropna()`, `.fillna()`, or by ensuring aggregation methods gracefully handle missing data.
        *   **Incorrect Data Types:** Explicitly convert column types using `.astype()` (e.g., `pd.to_numeric(df['col'], errors='coerce')`, `df['col'].astype(str)`, `pd.to_datetime(df['col'], errors='coerce')`) *before* performing numerical, date, or string operations if the inferred types in `df_types` might cause errors. Use `errors='coerce'` for numeric/datetime conversions to turn unparseable values into `NaN` instead of raising an error.
        *   **Key Errors:** Double-check column names against `df_preview` and `df_types` to avoid `KeyError` if a column name is slightly off.
    5.  **No External Code:** Your code MUST NOT include any `import` statements or `print()` calls. Only pure Pandas/Python logic is allowed.
    6.  **Output Format:** Enclose your entire Python code block within `<result></result>` XML tags.

    ## Example of Desired Output Structure:
    <result>
    # Example: Find the top 3 administrative districts by total '층수' (floor count)
    df['층수'] = pd.to_numeric(df['층수'], errors='coerce') # Ensure '층수' is numeric
    floor_counts = df.groupby('행정구')['층수'].sum().reset_index()
    sorted_districts = floor_counts.sort_values(by='층수', ascending=False)
    final_df = sorted_districts.head(3) # Provide contextual data, not just the top one name
    </result>
    """
    return prompt

def extract_code_from_response(response: str) -> str:
    """
    # 주석: LLM의 텍스트 응답에서 Python 코드 블록을 추출합니다.
    # 주석: <result> 태그 내부 또는 일반 마크다운 코드 블록에서 코드를 찾습니다.
    """
    # 1. Try to extract from <result> tags first (most reliable)
    match = re.search(r"<result>(.*?)</result>", response, re.DOTALL)
    if match:
        code_block = match.group(1)
        # 주석: <result> 태그 안에 마크다운 코드 펜스가 있는 경우 이를 제거합니다.
        code_block = re.sub(r"```(?:python)?", "", code_block).strip()
        return code_block

    # 2. If <result> tags are not found, try common markdown code blocks
    match = re.search(r"```(?:python)?(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 주석: 코드 추출 실패 시 경고 메시지를 표시합니다.
    st.warning("Could not extract any valid Python code from the LLM's response. Please ensure it's wrapped in `<result></result>` tags or markdown code blocks.")
    return ""

def execute_generated_code(code: str, df: pd.DataFrame, max_retries: int = 3) -> pd.DataFrame | str:
    """
    # 주석: 생성된 Python 코드를 실행하고, 오류가 발생하면 LLM에게 코드 수정을 요청하며 여러 번 재시도합니다.
    # 주석: 최종적으로 Pandas DataFrame을 반환하거나 오류 메시지 문자열을 반환합니다.
    """
    current_code = code
    error_history = [] # 주석: 이전 오류 메시지들을 기록합니다.
    
    st.info("Attempting to execute the generated Python code...") # 주석: 코드 실행 시도 알림

    for attempt in range(max_retries):
        try:
            # 주석: 코드 실행을 위한 로컬 변수 환경을 설정합니다.
            # 주석: df, final_df 외에 pd (pandas)와 np (numpy)를 미리 사용할 수 있도록 주입합니다.
            local_vars = {"df": df, "final_df": None, "pd": pd, "np": np}
            
            # 주석: 생성된 Python 코드를 실행합니다.
            # 주석: __builtins__=None은 내장 함수(예: print) 사용을 제한합니다.
            exec(current_code, {"__builtins__": None}, local_vars)
            
            # 주석: 실행 결과로 final_df가 유효한 Pandas DataFrame인지 확인합니다.
            result_df = local_vars.get("final_df")
            if isinstance(result_df, pd.DataFrame):
                st.success(f"Code executed successfully on attempt {attempt + 1}!") # 주석: 코드 실행 성공 메시지
                return result_df
            else:
                # 주석: final_df가 DataFrame이 아니거나 존재하지 않는 경우 오류를 발생시킵니다.
                raise ValueError("The generated code did not return a pandas DataFrame named `final_df` or `final_df` is not a DataFrame.")

        except Exception as e:
            error_message = str(e)
            error_history.append(f"Attempt {attempt + 1} failed with error: {error_message}")
            st.error(f"Code execution failed on attempt {attempt + 1}: `{error_message}`") # 주석: 코드 실행 실패 메시지
            
            if attempt < max_retries - 1:  # 주석: 마지막 시도가 아닌 경우에만 재시도 및 코드 수정 요청
                st.info(f"Requesting LLM to correct the code (attempt {attempt + 1}/{max_retries})...") # 주석: LLM에 코드 수정 요청 알림
                # 주석: 코드 수정을 위한 새로운 프롬프트를 생성합니다.
                # 주석: 이전 코드, 오류 메시지, 그리고 이전 오류 기록을 제공하여 LLM이 맥락을 이해하도록 돕습니다.
                error_prompt = f"""
                The following Python code failed during execution:
                ```python
                {current_code}
                ```
                
                The specific error message received was:
                {error_message}
                
                Here is the history of previous errors encountered with this query:
                {chr(10).join(error_history[:-1]) if error_history[:-1] else "No previous errors in this sequence."}

                **Please generate corrected Python code.** Keep the following in mind:
                1.  **Strictly adhere to the previous instructions:** No `import` statements, no `print()` calls, and the final result MUST be assigned to `final_df` as a Pandas DataFrame.
                2.  **Focus on the error:** Analyze the error message (`{error_message}`) to understand the root cause.
                3.  **Common fixes:**
                    *   **Type Conversion Issues:** If the error indicates a type mismatch (e.g., 'could not convert string to float', 'unsupported operand types'), use `pd.to_numeric(..., errors='coerce')`, `df['column'].astype(str)`, or `pd.to_datetime(..., errors='coerce')` for the relevant columns.
                    *   **Missing Column/KeyError:** Verify column names against the provided `df_preview` and `df_types`. Ensure you're using the exact column names.
                    *   **NaN Handling:** If calculations fail due to `NaN`s, consider `.dropna()`, `.fillna()`, or using aggregation methods that handle `NaN`s (e.g., `sum(skipna=True)`).
                    *   **Logic Errors:** If the error is not syntactic but logical (e.g., trying to average a non-numeric column after type conversion), re-evaluate the pandas operation.
                4.  **Avoid previous mistakes:** Review the `error_history` to ensure you don't repeat the same error.
                5.  **Output Format:** Wrap your corrected Python code within `<result></result>` XML tags.
                """
                
                # 주석: LLM에게 수정된 코드를 요청합니다.
                corrected_response = call_gemini_api(error_prompt)
                corrected_code = extract_code_from_response(corrected_response)
                
                if corrected_code and corrected_code != current_code: # 주석: 유효한 새 코드가 생성되었는지 확인
                    st.info("LLM provided a corrected code. Retrying with the new code.") # 주석: 수정된 코드 수신 알림
                    current_code = corrected_code
                else:
                    return f"Failed to get a corrected or different code from LLM after error: {error_message}. Error history: {chr(10).join(error_history)}"
            else:
                # 주석: 최대 재시도 횟수에 도달한 경우 오류를 반환합니다.
                return f"Maximum retry attempts ({max_retries}) reached. Failed to execute code. Error history: {chr(10).join(error_history)}"
    
    # 주석: 예기치 않은 오류가 발생한 경우 (이론적으로 여기에 도달해서는 안 됨)
    return f"An unexpected error occurred after all retries. Error history: {chr(10).join(error_history)}"


def generate_final_answer_prompt(user_query: str, filtered_df: pd.DataFrame) -> str:
    """
    # 주석: 필터링되거나 분석된 데이터를 기반으로 사용자 질의에 대한 최종 답변을 생성하기 위한 프롬프트를 만듭니다.
    # 주석: 이 프롬프트는 LLM이 주어진 데이터를 명확하고 간결하게 요약하도록 지시합니다.
    """
    # 주석: DataFrame을 JSON 문자열로 변환합니다. LLM이 데이터를 구조화된 형태로 이해하기 쉽도록 합니다.
    # 주석: orient="records"는 각 행을 객체로 표현하는 JSON 배열을 생성합니다.
    # 주석: force_ascii=False는 한국어 문자가 올바르게 인코딩되도록 합니다.
    try:
        if not filtered_df.empty:
            filtered_json = filtered_df.to_json(orient="records", force_ascii=False, indent=2)
        else:
            filtered_json = "[]" # 주석: DataFrame이 비어있으면 빈 배열을 보냅니다.
    except Exception as e:
        filtered_json = "[]" # 주석: JSON 변환 실패 시 비어있는 배열로 대체합니다.
        st.warning(f"Failed to convert `filtered_df` to JSON for final prompt: {e}. Sending an empty array.") # 주석: JSON 변환 실패 경고
    
    # 주석: 최종 답변 프롬프트 구성. LLM이 답변을 생성하는 방법에 대한 구체적인 지침을 포함합니다.
    prompt = f"""
    You are a helpful and concise data analysis assistant. Your role is to provide a clear and direct answer to the user's question based *solely* on the provided `Analyzed Data`.

    **Context:**
    - **User Query:** "{user_query}"
    - **Analyzed Data (JSON format):**
    ```json
    {filtered_json}
    ```

    **Instructions for generating the answer:**
    1.  **Direct Answer:** Provide a concise, clear, and direct answer to the user's query.
    2.  **Data-Driven:** Your answer must be entirely derived from the `Analyzed Data` provided. Do not use any external knowledge.
    3.  **Highlight Key Findings:** If applicable, summarize the most important insights or trends visible in the `Analyzed Data` that relate to the query.
    4.  **Handle Empty Data:** If the `Analyzed Data` is empty or does not contain sufficient information to answer the query, explicitly state that the answer cannot be determined from the provided data.
    5.  **Language:** The answer must be in Korean.
    6.  **Formatting:** Avoid unnecessary formatting, special characters, or code blocks in your final answer. Just plain, clear text.
    """
    return prompt

# --- Streamlit Application ---

def main():
    """
    # 주석: Streamlit 웹 애플리케이션의 메인 함수입니다.
    # 주석: 파일 업로드, 질문 입력, LLM 호출, 코드 실행 및 결과 표시의 전체 워크플로우를 관리합니다.
    """
    # 주석: Streamlit 페이지의 기본 설정 (전체 너비, 페이지 제목)
    st.set_page_config(layout="wide", page_title="AI Data Analysis Assistant")
    st.title("AI Data Analysis Assistant")
    st.markdown("Upload your Excel or CSV file and ask questions. The AI will generate and execute Python code to analyze your data and provide answers.")

    uploaded_file = st.file_uploader("Upload your data file (Excel or CSV)", type=["xls", "xlsx", "csv"])

    df: pd.DataFrame | None = None # 주석: 업로드된 파일을 저장할 DataFrame 변수를 초기화합니다.
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        st.info(f"Loading `{uploaded_file.name}` (type: {file_type})...") # 주석: 파일 로딩 시작 알림
        try:
            if file_type == 'csv':
                # 주석: CSV 파일 로드. 다양한 인코딩 시도 (utf-8이 실패할 경우).
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(uploaded_file, encoding='cp949') # 주석: 한국어 인코딩 시도
            else:
                # 주석: Excel 파일 로드.
                df = pd.read_excel(uploaded_file)
            st.success("File loaded successfully! 🎉") # 주석: 파일 로드 성공 알림
        except Exception as e:
            st.error(f"Error loading file: `{e}`. Please ensure it's a valid {file_type} file and try again.") # 주석: 파일 로드 오류 메시지
            return # 주석: 파일 로드 실패 시 함수를 종료합니다.

        # --- Display Data Previews for User and LLM ---
        with st.expander("👁️ Data Preview (for Human)"):
            st.dataframe(df.head(10)) # 주석: 사용자에게 처음 10개 행을 보여줍니다.

        # 주석: LLM에 보낼 데이터 미리보기 (처음 5개 행만)
        df_preview = df.head(5).to_dict(orient="records")
        with st.expander("🤖 Data Preview (for LLM - first 5 rows in JSON)"):
            st.json(df_preview)
        
        # 주석: LLM에 보낼 DataFrame의 각 컬럼 데이터 타입 정보
        df_types = df.dtypes.apply(lambda x: str(x)).to_dict()
        with st.expander("🤖 Data Types (for LLM in JSON)"):
            st.json(df_types)
        
        # --- Example Questions for User Convenience ---
        example_questions = [
            "What are the top 5 administrative districts by the total number of stores in Seoul, categorized by major business type?",
            "Which administrative district in Seoul has the highest average floor count for cafes?", 
            "In Seoul, which area has the highest proportion of real estate agencies out of all commercial stores?",
            "What is the proportion of stores by sub-business type (중분류) in Seongdong-gu?",
            "Show me the average floor count for all stores in Gangnam-gu, grouped by major business type.",
            "Can you give me the number of stores in each '행정구' (administrative district)?"
        ]

        # 주석: Streamlit 세션 상태를 사용하여 사용자 질의를 유지합니다.
        if "user_query" not in st.session_state:
            st.session_state["user_query"] = ""

        # --- User Input Area ---
        user_input = st.text_input(
            "📝 Ask a question about your data:",
            value=st.session_state["user_query"], # 주석: 이전에 입력된 질의를 유지합니다.
            key="input_box",
            help="Enter your question here or select from the examples below."
        )

        # 주석: 사용자가 입력이 없으면 예시 질문을 선택할 수 있도록 합니다.
        user_query_to_process = user_input
        if not user_input:
            sample_query = st.selectbox("Or select an example question:", [""] + example_questions, key="sample_box")
            user_query_to_process = sample_query

        # --- "Get Answer" Button ---
        if st.button("🚀 Get Answer", type="primary"):
            st.session_state["user_query"] = user_query_to_process # 주석: 현재 질의를 세션 상태에 저장합니다.
            if not user_query_to_process:
                st.warning("Please enter a question or select an example to get started.")
                return # 주석: 질의가 없으면 함수를 종료합니다.

            st.markdown("---")
            st.subheader("💡 Processing your request...")
            
            # 1. Generate Code Prompt for LLM
            # 주석: 사용자 질의와 데이터 메타데이터를 기반으로 LLM에게 코드 생성을 요청하는 프롬프트를 만듭니다.
            code_prompt = generate_code_prompt(user_query_to_process, df_preview, df_types)
            # with st.expander("📄 Generated Code Prompt (for LLM)"): # 주석: 디버깅을 위해 프롬프트 내용을 숨김/표시할 수 있습니다.
            #     st.code(code_prompt, language="markdown")

            # 2. Call LLM to Generate Code
            # 주석: Gemini API를 호출하여 Pandas 코드를 생성합니다.
            generated_response = call_gemini_api(code_prompt)
            if "ERROR" in generated_response:
                st.error("Failed to generate code from LLM due to an API error.")
                return

            # 주석: LLM 응답에서 실제 Python 코드 블록을 추출합니다.
            generated_code = extract_code_from_response(generated_response)
            if not generated_code:
                st.error("Could not extract Python code from LLM's response. Please try rephrasing your question or check the raw LLM response below.")
                with st.expander("Raw LLM Response"): # 주석: 코드 추출 실패 시 원본 응답을 보여줍니다.
                    st.code(generated_response, language="markdown", help="This is the raw output from the LLM.")
                return

            # 3. Execute Generated Code
            # 주석: 생성된 코드를 실행하고, 오류 발생 시 LLM에게 수정을 요청하며 재시도합니다.
            filtered_df_or_error = execute_generated_code(generated_code, df)
            
            if isinstance(filtered_df_or_error, pd.DataFrame):
                # 4. Generate Final Answer Prompt
                # 주석: 필터링된 데이터를 기반으로 최종 답변 생성을 위한 프롬프트를 만듭니다.
                final_answer_prompt = generate_final_answer_prompt(user_query_to_process, filtered_df_or_error)
                # with st.expander("📄 Final Answer Prompt (for LLM)"): # 주석: 디버깅을 위해 프롬프트 내용을 숨김/표시할 수 있습니다.
                #     st.code(final_answer_prompt, language="json")
                
                # 5. Call LLM for Final Answer
                # 주석: Gemini API를 호출하여 최종 자연어 답변을 생성합니다.
                final_response = call_gemini_api(final_answer_prompt)
                if "ERROR" in final_response:
                    st.error("Failed to generate final answer from LLM.")
                    return

                # --- Display Results to User ---
                st.subheader("✅ Answer")
                st.success(final_response) # 주석: 최종 답변을 사용자에게 표시합니다.
                
                with st.expander("🔍 Detailed Analysis (for debugging and transparency)"):
                    st.markdown("---")
                    st.subheader("Generated Python Code")
                    st.code(generated_code, language="python")

                    st.subheader("Intermediate Filtered/Analyzed Data") 
                    if not filtered_df_or_error.empty:
                        st.dataframe(filtered_df_or_error)
                    else:
                        st.info("The generated code resulted in an empty DataFrame. This might mean no data matched your query or the analysis produced no results.")
                                                
                    st.subheader("LLM Prompt used for Final Answer Generation")
                    st.code(final_answer_prompt, language="json")

            else:
                # 주석: 코드 실행 중 오류가 발생한 경우 오류 메시지를 표시합니다.
                st.error(f"An error occurred during code execution that could not be resolved: {filtered_df_or_error}")

if __name__ == "__main__":
    main()