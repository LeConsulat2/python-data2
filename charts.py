import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 데이터 준비: 사용자가 제공한 데이터를 기반으로 샘플 데이터프레임 생성
data = {
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

reason_df = pd.DataFrame(data)

# 설정
plt.figure(figsize=(16, 12))
sns.set_theme(style="whitegrid")

# 중앙 제목
plt.suptitle(
    "The Top Three Reasons for M&A Account for 47.6% of the Total Across the 10 Categories",
    fontsize=18,
    fontweight="bold",
    x=0.5,
    y=0.92,
)

# 1. M&A 이유 막대그래프
plt.subplot(2, 2, 1)
sns.barplot(data=reason_df, x="Count", y="Reason", palette="Blues_r")
plt.title("Top Reasons for M&A", fontsize=14)
plt.xlabel("Count")
plt.ylabel("Reason")

# 2. 연도별 M&A 활동량 그래프 (샘플 데이터)
years = ["2018", "2019", "2020", "2021", "2022"]
activity = [61, 58, 60, 65, 77]

plt.subplot(2, 2, 2)
sns.lineplot(x=years, y=activity, marker="o", color="blue")
plt.title("Annual Corporate M&A Activities", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Activity Count")

# 3. M&A 거래 비율 버블 차트
plt.subplot(2, 2, 3)
bubble_sizes = np.array(reason_df["Count"]) * 5
plt.scatter(
    reason_df["Count"],
    range(len(reason_df)),
    s=bubble_sizes,
    alpha=0.6,
    c=bubble_sizes,
    cmap="Blues",
)
plt.title("M&A Transactions Contribution", fontsize=14)
plt.xlabel("M&A Deals")
plt.yticks(range(len(reason_df)), reason_df["Reason"])
plt.ylabel("Reasons")

# 4. 도넛 차트
sectors = ["Construction", "Industrial", "Technology", "Energy"]
values = [583, 557, 273, 247]

plt.subplot(2, 2, 4)
wedges, texts, autotexts = plt.pie(
    values,
    labels=sectors,
    autopct="%1.1f%%",
    startangle=140,
    colors=sns.color_palette("pastel"),
    wedgeprops=dict(width=0.3),
)
plt.title("Sector Contributions to M&A", fontsize=14)

# 전체 레이아웃 조정
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()
