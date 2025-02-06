import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


############### TO  SEE THE RESULT, PLEASE REFER TO Figure_4.png ###############
############### TO  SEE THE RESULT, PLEASE REFER TO Figure_4.png ###############
############### TO  SEE THE RESULT, PLEASE REFER TO Figure_4.png ###############
############### TO  SEE THE RESULT, PLEASE REFER TO Figure_4.png ###############

# Set style properly
# plt.style.use("seaborn-v0_8")
# sns.set_palette("Blues_r")

# 흰 배경 + 스타일 설정
# sns.set_style("whitegrid")
# fig = plt.figure(figsize=(20, 14))  20 가로 / 14 세로로

############### 여기부터 다크 테마 설정 추가 ###############
# 테마 및 스타일 설정 다크 = dark_background
plt.style.use("seaborn-v0_8-notebook")
sns.set_palette("Blues_r")
fig = plt.figure(figsize=(16, 10))  # 16 가로 / 10 세로로
fig.patch.set_facecolor("#0E1117")  # 전체 배경


"""
fig는 matplotlib의 Figure 객체입니다:
fig = plt.figure()는 matplotlib.pyplot (plt)에서 새로운 그래프 창을 생성
plt는 matplotlib.pyplot의 약자
Figure는 전체 그래프 창을 의미하며, 여러 개의 서브플롯(subplot)을 포함할 수 있음
plt.rcParams는 matplotlib의 런타임 설정(runtime configuration parameters)입니다:
rc는 "runtime configuration"의 약자
"""

# 스타일 파라미터 설정
plt.rcParams["axes.facecolor"] = "#0E1117"  # 그래프 배경
plt.rcParams["text.color"] = "white"  # 모든 텍스트 색상
plt.rcParams["axes.labelcolor"] = "white"  # 축 레이블 색상
plt.rcParams["xtick.color"] = "white"  # x축 눈금 색상
plt.rcParams["ytick.color"] = "white"  # y축 눈금 색상
############### 다크 테마 설정 끝 ###############

# Data preparation (same as before)
reasons_data = {
    "Reason": [
        "Cost Synergy",
        "Market Expansion",
        "Acquisition of Talent",
        "Vertical Integration",
        "Acquisition of Technology",
        "Revenue Synergy",
        "Elimination of Competitors",
        "Product Diversification",
        "Financial Improvement",
        "Strategic Restructuring",
    ],
    "Count": [206, 169, 150, 127, 116, 90, 72, 69, 54, 50],
}

# Data for country comparison
countries = ["China", "Germany", "India", "Korea", "USA"]
years = [2018, 2019, 2020, 2021, 2022]
data_by_country = {
    "China": [30, 27, 39, 40, 43],
    "Germany": [18, 15, 13, 24, 25],
    "India": [19, 13, 13, 25, 25],
    "Korea": [29, 21, 33, 54, 52],
    "USA": [40, 42, 47, 24, 66],
}


# Create figure with more height and adjust spacing
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Increase spacing between subplots

# 메인 타이틀 위치 조정
fig.suptitle(
    "M&A Analysis Overview", fontsize=16, fontweight="bold", y=0.98
)  # y 값을 0.95에서 0.98로 올림

# 전체 레이아웃 조정
plt.subplots_adjust(
    top=0.90,  # top 값을 0.85에서 0.90으로 수정 (서브플롯을 아래로)
    bottom=0.15,
    left=0.2,
    right=0.87,
    wspace=0.3,
    hspace=0.4,
)

# 1. Top reasons bar chart
ax1 = plt.subplot(2, 2, 1)
reasons_df = pd.DataFrame(reasons_data)
bars = sns.barplot(
    data=reasons_df, x="Count", y="Reason", color="#1f77b4"
)  # x="Count", y="Reason",
plt.title("Top 10 Reasons for M&A", pad=20, fontsize=14, fontweight="bold")

# 축 레이블 제거
plt.xlabel("")
plt.ylabel("")
# x축 레이블 회전 없음
plt.xticks(rotation=0)


# Add value labels on the bars
for i, v in enumerate(reasons_df["Count"]):
    ax1.text(v + 1, i, str(v), va="center")

# 2. Country comparison chart
ax2 = plt.subplot(2, 2, 2)
x = np.arange(len(countries))
width = 0.15
# x축 레이블 45도 회전
# plt.xticks(rotation=45, ha="left")

for i, year in enumerate(years):
    data = [data_by_country[country][i] for country in countries]
    plt.bar(x + i * width, data, width, label=str(year))

# plt.xlabel("Countries")
plt.ylabel("Number of M&A Activities")
plt.title("M&A Activities by Country and Year", pad=20, fontsize=14, fontweight="bold")
plt.xticks(x + width * 2, countries)
plt.legend(bbox_to_anchor=(1.15, 1), loc="upper right")

# 3. Quarterly trends
ax3 = plt.subplot(2, 2, 3)
quarters = ["Q1", "Q2", "Q3", "Q4"]
years = [2018, 2019, 2020, 2021, 2022]
# x축 레이블회전없음
plt.xticks(rotation=0)

quarterly_data = {
    2018: [25, 35, 30, 40],
    2019: [32, 38, 35, 42],
    2020: [28, 45, 38, 48],
    2021: [35, 50, 45, 55],
    2022: [40, 55, 50, 60],
}

for year in years:
    plt.plot(quarters, quarterly_data[year], marker="o", label=str(year))

plt.title("Quarterly M&A Trends", pad=20, fontsize=14, fontweight="bold")
plt.xlabel("Quarter")
plt.ylabel("Number of Deals")
plt.legend(
    bbox_to_anchor=(-0.30, 1), loc="upper left"
)  # 0.05는 왼쪽 5% 위치, loc를 upper left로 변경
plt.grid(True)


# 4. Sector distribution donut chart
ax4 = plt.subplot(
    2, 2, 4, position=(0.55, 0.2, 0.4, 0.4)
)  # [left, bottom, width, height]
sectors = [
    "Construction Machinery",
    "Industrial Machinery",
    "Agricultural Machinery",
    "Mining Equipment",
    "Finance",
    "Technology",
    "Energy",
]
sector_values = [583, 557, 396, 247, 273, 273, 247]

plt.pie(
    sector_values,
    labels=sectors,
    autopct="%1.1f%%",
    startangle=90,
    pctdistance=0.85,
    wedgeprops=dict(width=0.5),
)
plt.title("Sector Distribution", pad=20, fontsize=14, fontweight="bold")

# Adjust layout with more space
plt.tight_layout(
    rect=[0, 0.03, 1, 0.95]
)  # Adjust the rect parameter to give more space for titles
plt.show()
