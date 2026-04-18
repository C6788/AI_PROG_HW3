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
# 任务 3：线路站点分析
print("\n=== 任务3：线路站点分析 ===")

#严格遵循函数签名规范
def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):
    """
    计算各线路乘客的平均搭乘站点数及其标准差。
    Parameters
    ----------
    df : pd.DataFrame  预处理后的数据集
    route_col : str    线路号列名
    stops_col : str    搭乘站点数列名
    Returns
    -------
    pd.DataFrame  包含列：线路号、mean_stops、std_stops，按 mean_stops 降序排列
    """
    #函数注释
    # 聚合计算均值和标准差
    stats_df = df.groupby(route_col)[stops_col].agg(['mean', 'std']).reset_index()
    # 重命名输出列
    stats_df.columns = [route_col, 'mean_stops', 'std_stops']
    # 按照平均搭乘站点数降序排序
    return stats_df.sort_values(by='mean_stops', ascending=False)

# 调用函数
route_stats = analyze_route_stops(df)
print("【各线路搭乘站点数统计（前10行）】：")
print(route_stats.head(10))

# 筛选均值最高的前 15 条线路进行可视化
top15_routes = route_stats.head(15).copy()
# 将线路号转为字符串，防止被识别为连续数值坐标
top15_routes['线路号'] = top15_routes['线路号'].astype(str)

plt.figure(figsize=(10, 8))
#seaborn制图
ax = sns.barplot(
    x='mean_stops',
    y='线路号',
    data=top15_routes,
    orient='h',
    palette="Blues_d",
    order=top15_routes['线路号']
)

# 增加误差棒
plt.errorbar(
    x=top15_routes['mean_stops'],
    y=np.arange(len(top15_routes)),
    xerr=top15_routes['std_stops'],
    fmt='none',           # 不绘制连线或标记点
    c='black',            # 误差棒颜色
    capsize=0.3 * 10,     # capsize=0.3在matplotlib绝对数值体系偏小，这里适当放大确保可见度
    elinewidth=1.2
)

plt.title('均值最高的前15条线路：平均搭乘站点数', fontsize=14)
plt.xlabel('平均搭乘站点数（附标准差）', fontsize=12)
plt.ylabel('线路号', fontsize=12)
plt.xlim(left=0)

plt.savefig('route_stops.png', dpi=150)
plt.close()
print("已成功生成图像：route_stops.png")
# 任务 4：高峰小时系数计算

print("\n=== 任务4：高峰小时系数计算 ===")

# 仅针对上车刷卡的记录进行计算
boarding_df = df[df['刷卡类型'] == 0].copy()

# 1. 自动识别高峰小时
hour_counts = boarding_df['hour'].value_counts()
peak_hour = hour_counts.idxmax()
peak_hour_vol = hour_counts.max()

peak_start = f"{peak_hour:02d}:00"
peak_end = f"{peak_hour+1:02d}:00"

# 取出高峰小时的完整切片
peak_df = boarding_df[boarding_df['hour'] == peak_hour].copy()


# 设置“交易时间”为 DataFrame 的索引，因为 pandas 的 resample(重采样) 方法必须依赖于 DatetimeIndex。
peak_df.set_index('交易时间', inplace=True)

# 使用 .resample('5min').size() 按照 5 分钟为一个时间窗口，对所有数据进行分桶聚合，并计算每个桶里的记录行数。
resampled_5m = peak_df.resample('5min').size()
# 从所有 5 分钟的聚合窗口中寻找极大值，以及该极大值对应的时间段起点。
max_5m_vol = resampled_5m.max()
max_5m_time = resampled_5m.idxmax()

# 计算 PHF5。公式分母的 12 是因为 1 小时等于 12 个 5 分钟窗口。
phf5 = peak_hour_vol / (12 * max_5m_vol)

# 同理，执行 15 分钟粒度的重采样聚合。
resampled_15m = peak_df.resample('15min').size()
max_15m_vol = resampled_15m.max()
max_15m_time = resampled_15m.idxmax()

# 计算 PHF15。分母的 4 代表 1 小时等于 4 个 15 分钟窗口。
phf15 = peak_hour_vol / (4 * max_15m_vol)

# 打印最终格式化结果
print(f"高峰小时：{peak_start} ~ {peak_end}，刷卡量：{peak_hour_vol} 次")
print(f"最大5分钟刷卡量（{max_5m_time.strftime('%H:%M')}~{(max_5m_time + pd.Timedelta(minutes=5)).strftime('%H:%M')}）：{max_5m_vol} 次")
print(f"PHF5  = {peak_hour_vol} / (12 × {max_5m_vol}) = {phf5:.4f}")
print(f"最大15分钟刷卡量（{max_15m_time.strftime('%H:%M')}~{(max_15m_time + pd.Timedelta(minutes=15)).strftime('%H:%M')}）：{max_15m_vol} 次")
print(f"PHF15 = {peak_hour_vol} / ( 4 × {max_15m_vol}) = {phf15:.4f}")
# 任务 5：线路驾驶员信息批量导出
print("\n=== 任务5：线路驾驶员信息批量导出 ===")

# 筛选目标线路 (1101 至 1120)
target_routes = df[(df['线路号'] >= 1101) & (df['线路号'] <= 1120)]

folder_name = '线路驾驶员信息'
# 若根目录下无此文件夹，则自动创建
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# 提取并排序这 20 条唯一线路
unique_routes = sorted(target_routes['线路号'].unique())

for route in unique_routes:
    # 提取特定线路，选取两列后利用 drop_duplicates() 去重
    pairs = target_routes[target_routes['线路号'] == route][['车辆编号', '驾驶员编号']].drop_duplicates()

    file_path = os.path.join(folder_name, f"{int(route)}.txt")

    # 写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"线路号: {int(route)}\n")
        f.write("车辆编号\t驾驶员编号\n")
        for _, row in pairs.iterrows():
            f.write(f"{int(row['车辆编号'])}\t{int(row['驾驶员编号'])}\n")

    print(f"已成功导出：{file_path} (共 {len(pairs)} 对关系)")

print("20个文件全部生成完毕！")
# 任务 6：服务绩效排名与热力图
print("\n=== 任务6：服务绩效排名与热力图 ===")

# 统计服务频次排名前 10 的各实体（基于搭乘乘客人次的有效行数）
top10_driver = df['驾驶员编号'].value_counts().head(10)
top10_route = df['线路号'].value_counts().head(10)
top10_stop = df['上车站点'].value_counts().head(10)
top10_vehicle = df['车辆编号'].value_counts().head(10)

print("【Top 10 司机】:\n", top10_driver.index.tolist())
print("【Top 10 线路】:\n", top10_route.index.tolist())
print("【Top 10 上车站点】:\n", top10_stop.index.tolist())
print("【Top 10 车辆】:\n", top10_vehicle.index.tolist())

# 构建供热力图读取的数据集
# 先用字典构建各列，再调用 .T 转置，使其成为 4 行  x 10 列
heatmap_data = pd.DataFrame({
    '司机': top10_driver.values,
    '线路': top10_route.values,
    '上车站点': top10_stop.values,
    '车辆': top10_vehicle.values
}).T

heatmap_data.columns = [f"Top {i}" for i in range(1, 11)]

plt.figure(figsize=(14, 5))
# 绘制 seaborn 热力图
sns.heatmap(
    heatmap_data,
    annot=True,      # 在格中标注数值
    fmt='g',         # 通用数字格式，防止出现不直观的科学计数法
    cmap="YlOrRd",   # 要求的 黄-橙-红过渡
    linewidths=1,
    linecolor='white'
)

plt.title('服务绩效排名 Top10 实体服务人次热力图', fontsize=16, pad=15)
plt.suptitle('衡量标准：各实体累计提供的搭乘有效上车刷卡人次', fontsize=11, color='gray', y=0.92)
# 保持坐标轴标签平直不旋转
plt.xticks(rotation=0)
plt.yticks(rotation=0)

# 使用 bbox_inches='tight' 防止图片边缘或标题被切除
plt.savefig('performance_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("已成功生成图像：performance_heatmap.png")

# 结论说明输出 (满足字数 > 50 字)
conclusion = """
【服务绩效热力图规律观察】：
通过观察 YlOrRd 色彩映射的热力图，我发现明显的结构性差异现象：在“线路”与“上车站点”这两个空间维度中，
Top 1 到 Top 3 的服务人次数值呈现出极深的红色，数值断层式领先于后排实体，表现出强烈的客流马太效应和枢纽汇聚特征。
相较之下，“司机”和“车辆”这两项由于物理运力与排班时间的刚性限制，其 Top 10 数值的色彩渐变非常平缓，未出现明显断层。
"""
print(conclusion)