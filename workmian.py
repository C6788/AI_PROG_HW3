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