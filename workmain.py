import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] # Windows 优先用 SimHei，Mac 优先用 Arial Unicode MS
matplotlib.rcParams['axes.unicode_minus'] = False

# 任务 1：数据预处理
print("=== 任务1：数据预处理 ===")

# 读取数据：既然真实数据是逗号分隔的，我们就把 sep='\t' 改为 sep=','（或者直接不写，默认就是逗号）
df = pd.read_csv('ICData.csv', sep=',')

print("【数据集前5行】：")
print(df.head())
print("\n【数据集基本信息】：")
df.info()

# 在 format 中加上秒的解析 '%S'
df['交易时间'] = pd.to_datetime(df['交易时间'], format='%Y/%m/%d %H:%M:%S')
df['hour'] = df['交易时间'].dt.hour

# 计算搭乘站点数并取绝对值
df['ride_stops'] = (df['下车站点'] - df['上车站点']).abs()

# 筛选出 ride_stops 大于 0 的记录
original_len = len(df)
df = df[df['ride_stops'] > 0].copy()
deleted_count = original_len - len(df)
print(f"\n【异常值处理】：删除了 {deleted_count} 行 ride_stops 为 0 的异常记录。")

# 检查与处理缺失值
print("\n【各列缺失值数量】：")
print(df.isnull().sum())
if df.isnull().sum().sum() > 0:
    # 策略说明：刷卡核心数据缺失难以推断填补，因此采取直接删除含缺失值的行记录
    df.dropna(inplace=True)
    print("【缺失值处理】：已执行 dropna() 删除包含缺失值的记录。")
else:
    print("【缺失值处理】：数据集中不存在缺失值，无需处理。")
# 任务 2：时间分布分析
print("\n=== 任务2：时间分布分析 ===")

# (a) 早晚时段刷卡量统计
# 提取 numpy 数组加速运算
is_boarding = df['刷卡类型'].values == 0
hours_arr = df['hour'].values

# 使用组合条件
early_mask = (hours_arr < 7) & is_boarding
late_mask = (hours_arr >= 22) & is_boarding

early_count = np.sum(early_mask)
late_count = np.sum(late_mask)
total_boarding = np.sum(is_boarding)

print(f"早峰前时段（<7点）刷卡量: {early_count} 次，占比: {(early_count / total_boarding) * 100:.2f}%")
print(f"深夜时段（>=22点）刷卡量: {late_count} 次，占比: {(late_count / total_boarding) * 100:.2f}%")

# (b) 24小时刷卡量分布可视化
# 统计每小时上车人数并补齐 0-23 小时，防止格式混乱
hourly_counts = df[df['刷卡类型'] == 0]['hour'].value_counts().sort_index()
all_hours = pd.Series(0, index=np.arange(24))
hourly_counts = hourly_counts.combine_first(all_hours)

plt.figure(figsize=(10, 6))
bars = plt.bar(hourly_counts.index, hourly_counts.values, color='steelblue', label='常规时段')

# 针对早峰前和深夜进行单独高亮变色
for i in range(24):
    if i < 7 or i >= 22:
        bars[i].set_color('salmon')
        if i == 0:  # 防止图例重复，只在第一个满足条件的柱子上打标签
            bars[i].set_label('早峰前 / 深夜时段')

# 图表装饰
plt.title('24小时上车刷卡量分布图', fontsize=14)
plt.xlabel('小时', fontsize=12)
plt.ylabel('刷卡量（次）', fontsize=12)
plt.xticks(np.arange(0, 24, 2))  # x 轴步长为 2
plt.grid(axis='y', linestyle='--', alpha=0.7)  # 水平网格线
plt.legend()

plt.savefig('hour_distribution.png', dpi=150)
plt.close()
print("已成功生成图像：hour_distribution.png")