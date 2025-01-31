import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

############### 첫 번째 create_visualizations() 함수 완전 삭제 ###############


def create_visualizations():
    # Table sheet만 로드
    df_table = pd.read_excel(
        "University_Data.xlsx",
        sheet_name="Table",
        engine="openpyxl",
    )

    # Duration 계산 (End Date - Start Date)
    df_table["Duration"] = df_table["End Date"] - df_table["Start Date"]

    ############### 여기부터 다크 테마 설정 추가 ###############
    # 다크 테마 및 스타일 설정
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#0E1117")  # 전체 배경

    # 흰 배경
    # 스타일 설정
    # sns.set_style("whitegrid")
    # fig = plt.figure(figsize=(20, 14))  # 세로 크기 증가

    # 스타일 파라미터 설정
    plt.rcParams["axes.facecolor"] = "#0E1117"  # 그래프 배경
    plt.rcParams["text.color"] = "white"  # 텍스트 색상
    plt.rcParams["axes.labelcolor"] = "white"  # 축 레이블 색상
    plt.rcParams["xtick.color"] = "white"  # x축 눈금 색상
    plt.rcParams["ytick.color"] = "white"  # y축 눈금 색상
    ############### 다크 테마 설정 끝 ###############

    # 메인 타이틀 추가
    fig.suptitle(
        "Analysis of University Programme Distribution",
        fontsize=24,
        fontweight="bold",
        y=0.95,
    )

    # 1. Bar Chart (상위 15개 학과만) - 가로 버전
    plt.subplot(2, 2, 1)
    dept_counts = df_table["Department"].value_counts().head(10)
    sns.barplot(y=dept_counts.index, x=dept_counts.values, palette="viridis")
    plt.title("Top 10 Programmers by Department", pad=20, fontsize=12)
    # plt.xlabel("Count", fontsize=10)
    plt.ylabel("Department", fontsize=10)

    # 2. Pie Chart와 데이터 테이블
    plt.subplot(2, 2, 2)
    qual_counts = df_table["Qualification Level"].value_counts()
    plt.pie(
        qual_counts,
        labels=qual_counts.index,
        autopct="%1.1f%%",
        colors=sns.color_palette("pastel"),
    )
    plt.title("Distribution of Qualification Levels", pad=20, fontsize=12)

    # Department별 Total Size 평균 계산 (상위 8개)
    dept_size = (
        df_table.groupby("Department")["Total Size"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    # 1. 테이블 헤더 생성
    table_data = [["Department", "Avg Size"]]  # 첫 번째 행은 컬럼 제목

    # 2. 각 학과별 데이터 추가
    for dept, size in dept_size.items():
        # dept_size는 학과별 평균 크기를 담고 있는 Series

        # 3. 긴 학과명 처리
        if len(dept) > 20:  # 학과명 즉 단어 갯수가가 20자보다 길면
            dept = dept[:20] + "..."  # 20자로 자르고 "..." 추가

        # 4. 데이터 행 추가
        table_data.append([dept, f"{size:.1f}"])  # size를 소수점 1자리로 포맷팅

    # 테이블 추가 (위치와 크기 조정)
    # bbox 방식 -> [left, bottom, width, height]
    table = plt.table(
        cellText=table_data,
        loc="right",
        bbox=[1.5, 0.0, 1.5, 1.5],
        cellLoc="left",
    )

    # 테이블 스타일 조정
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(2.5, 1.3)

    # 테이블 색상 설정
    for cell in table._cells.values():
        cell.set_facecolor("#0E1117")  # 배경색을 다크 모드와 동일하게
        cell.set_text_props(color="white")  # 텍스트 색상을 흰색으로
        cell.set_edgecolor("white")  # 테두리 색상을 흰색으로

    # 3. Program Duration 분포 시각화
    plt.subplot(2, 2, 3)
    qual_durations = (
        df_table.groupby("Qualification Level")["Duration"]
        .mean()
        .sort_values(ascending=False)
    )
    sns.barplot(x=qual_durations.index, y=qual_durations.values, palette="cool")
    plt.title("Average Duration by Qualification Level", pad=20, fontsize=12)
    plt.xlabel("Qualification Level", fontsize=10)
    plt.ylabel("Average Duration", fontsize=10)

    # Duration 값 표시
    for i, v in enumerate(qual_durations.values):
        plt.text(x=i, y=v, s=int(v), ha="center", va="bottom")

    # 4. Bar Plot (Total Size)
    plt.subplot(2, 2, 4)
    qual_sizes = (
        df_table.groupby("Qualification Level")["Total Size"]
        .mean()
        .sort_values(ascending=False)
    )
    sns.barplot(x=qual_sizes.index, y=qual_sizes.values, palette="cool")
    plt.title("Average Total Size by Qualification Level", pad=20, fontsize=12)
    plt.xlabel("Qualification Level", fontsize=10)
    plt.ylabel("Average Total Size", fontsize=10)

    # Total Size 값 표시
    for i, v in enumerate(qual_sizes.values):
        plt.text(x=i, y=v, s=int(v), ha="center", va="bottom")

    # 전체 레이아웃 조정
    plt.subplots_adjust(
        top=0.80,
        bottom=0.1,
        left=0.2,
        right=0.8,
        wspace=0.3,
        hspace=0.4,
    )
    plt.show()


if __name__ == "__main__":
    create_visualizations()
