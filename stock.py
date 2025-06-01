import requests
import pandas as pd


def fetch_page(p):
    """
    p 페이지를 호출해서 DataFrame 으로 변환
    """
    url = "https://api.stock.naver.com/stock/exchange/NASDAQ/marketValue"
    params = {"page": p, "pageSize": 100}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()["stocks"]

    # JSON 리스트 → DataFrame
    df = pd.json_normalize(data)

    # industryCodeType 안의 industryGroupKor 꺼내서 새 컬럼으로
    if "industryCodeType.industryGroupKor" in df.columns:
        df["industryGroupKor"] = df["industryCodeType.industryGroupKor"]

    # 필요한 컬럼만 선택
    cols = [
        "reutersCode",
        "symbolCode",
        "stockName",
        "stockNameEng",
        "industryGroupKor",
        "openPrice",
        "closePrice",
        "fluctuationsRatio",
        "compareToPreviousClosePrice",
        "accumulatedTradingVolume",
        "marketValue",
        "endUrl",
    ]
    return df[cols]


def main():
    # 1) 총 페이지 수를 동적으로 구하려면, 첫 페이지 호출해서 API 응답 내 totalPages 필드 확인
    first = requests.get(
        "https://api.stock.naver.com/stock/exchange/NASDAQ/marketValue",
        params={"page": 1, "pageSize": 100},
    ).json()
    total_pages = first.get("totalPages", 10)  # API 문서대로 필드 이름 바꿔주세요

    # 2) 모든 페이지 fetch
    all_dfs = [fetch_page(p) for p in range(1, total_pages + 1)]

    # 3) 하나의 DataFrame으로 합치기
    df = pd.concat(all_dfs, ignore_index=True)

    # 4) 파일로 저장
    df.to_excel("nasdaq_market_value.xlsx", index=False)
    print("저장 완료: nasdaq_market_value.xlsx")


if __name__ == "__main__":
    main()
