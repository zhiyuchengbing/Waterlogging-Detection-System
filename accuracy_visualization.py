#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
列车积水识别准确率数据可视化工具
集成到主系统中的准确率可视化模块
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 固定 data 目录为绝对路径，保持与主程序一致
DATA_DIR = r"E:\积水识别项目\demo0625\demo\data"

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 尝试解决中文字体问题的额外设置
try:
    import matplotlib.font_manager as fm
    # 获取系统可用字体
    font_list = [f.name for f in fm.fontManager.ttflist]
    chinese_fonts = [f for f in font_list if any(ch in f for ch in ['Microsoft YaHei', 'SimHei', '微软雅黑', '黑体'])]
    if chinese_fonts:
        plt.rcParams['font.family'] = chinese_fonts[0]
        print(f"使用中文字体: {chinese_fonts[0]}")
except:
    print("使用默认字体设置")

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QDateEdit, QTextEdit, QGroupBox, 
    QMessageBox, QStatusBar, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import Qt, QDate, QThread, Signal, QTimer
from PySide6.QtGui import QFont


class TrainAccuracyMainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.data = None
        self.current_stats = None
        
        self.setWindowTitle("列车积水识别准确率数据可视化系统")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet(self.get_style_sheet())
        
        self.setup_ui()
        self.setup_status_bar()
        
        # 自动加载数据
        QTimer.singleShot(500, self.load_data)
    
    def get_style_sheet(self):
        """获取样式表"""
        return """
        QMainWindow { background-color: #f5f5f5; }
        QGroupBox {
            font-weight: bold; border: 2px solid #cccccc; border-radius: 8px;
            margin-top: 1ex; padding-top: 10px; background-color: white;
        }
        QPushButton {
            background-color: #3498db; border: none; color: white;
            padding: 8px 16px; border-radius: 6px; font-weight: bold; min-width: 100px;
        }
        QPushButton:hover { background-color: #2980b9; }
        QPushButton:disabled { background-color: #bdc3c7; }
        QDateEdit {
            padding: 5px; border: 2px solid #bdc3c7; 
            border-radius: 4px; background-color: white;
        }
        QTextEdit {
            border: 2px solid #bdc3c7; border-radius: 4px; 
            background-color: white; font-family: monospace; font-size: 10pt;
        }
        """
    
    def setup_ui(self):
        """设置用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        
        # 控制面板
        control_group = QGroupBox("控制面板")
        control_layout = QHBoxLayout(control_group)
        
        control_layout.addWidget(QLabel("开始日期:"))
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setDate(QDate(2025, 6, 1))
        self.start_date_edit.setCalendarPopup(True)
        control_layout.addWidget(self.start_date_edit)
        
        control_layout.addWidget(QLabel("结束日期:"))
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setDate(QDate(2025, 9, 15))
        self.end_date_edit.setCalendarPopup(True)
        control_layout.addWidget(self.end_date_edit)
        
        control_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        self.load_btn = QPushButton("加载数据")
        self.load_btn.clicked.connect(self.load_data)
        control_layout.addWidget(self.load_btn)
        
        self.analyze_btn = QPushButton("生成分析")
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setEnabled(False)
        control_layout.addWidget(self.analyze_btn)
        
        self.chart_btn = QPushButton("显示图表")
        self.chart_btn.clicked.connect(self.show_charts)
        self.chart_btn.setEnabled(False)
        control_layout.addWidget(self.chart_btn)
        
        main_layout.addWidget(control_group)
        
        # 信息面板
        info_group = QGroupBox("统计信息")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        info_layout.addWidget(self.info_text)
        
        main_layout.addWidget(info_group)
    
    def setup_status_bar(self):
        """设置状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("准备就绪")
    
    def load_data(self):
        """加载数据"""
        try:
            self.status_bar.showMessage("正在加载数据...")
            
            # 尝试多个可能的文件路径（优先绝对路径）
            possible_paths = [
                DATA_DIR + "/rate.csv",
                "data/rate.csv",
                "./data/rate.csv",
                "rate.csv",
                "data\\rate.csv",
                ".\\data\\rate.csv"
            ]
            
            data_loaded = False
            for path in possible_paths:
                try:
                    self.data = pd.read_csv(path, encoding='utf-8')
                    data_loaded = True
                    break
                except FileNotFoundError:
                    continue
                except Exception as e:
                    print(f"尝试路径 {path} 时出错: {e}")
                    continue
            
            if not data_loaded:
                raise FileNotFoundError("无法找到 rate.csv 文件，请确保文件位于以下位置之一：data/rate.csv, ./data/rate.csv, rate.csv")
            
            # 从列车名称中提取日期时间
            try:
                # 列车名称格式：YYYYMMDDHHMMSS (14位数字)
                # 例如：20250915071121 表示 2025-09-15 07:11:21
                train_names = self.data['列车名称'].astype(str)
                
                # 确保所有列车名称都是14位数字
                valid_names = train_names.str.len() == 14
                if not valid_names.all():
                    invalid_count = (~valid_names).sum()
                    print(f"警告：发现 {invalid_count} 个格式不正确的列车名称，将跳过这些记录")
                    self.data = self.data[valid_names].copy()
                    train_names = self.data['列车名称'].astype(str)
                
                # 提取日期时间部分
                years = train_names.str[:4]
                months = train_names.str[4:6] 
                days = train_names.str[6:8]
                hours = train_names.str[8:10]
                minutes = train_names.str[10:12]
                seconds = train_names.str[12:14]
                
                # 组合成标准日期时间格式
                datetime_strings = years + '-' + months + '-' + days + ' ' + hours + ':' + minutes + ':' + seconds
                
                # 转换为datetime对象
                self.data['日期时间'] = pd.to_datetime(datetime_strings, format='%Y-%m-%d %H:%M:%S')
                
            except Exception as e:
                raise ValueError(f"日期时间解析失败：{e}\n请检查列车名称格式是否为 YYYYMMDDHHMMSS (14位数字)")
            
            # 提取日期
            self.data['日期'] = self.data['日期时间'].dt.date
            
            # 数据类型转换
            try:
                self.data['识别正确数'] = pd.to_numeric(self.data['识别正确数'], errors='coerce')
                self.data['参与统计数'] = pd.to_numeric(self.data['参与统计数'], errors='coerce')
                self.data['总车厢数'] = pd.to_numeric(self.data['总车厢数'], errors='coerce')
                self.data['准确率(%)'] = pd.to_numeric(self.data['准确率(%)'], errors='coerce')
                
                # 检查是否有缺失值
                if self.data[['识别正确数', '参与统计数', '总车厢数']].isnull().any().any():
                    print("警告：数据中存在缺失值，将用 0 填充")
                    self.data[['识别正确数', '参与统计数', '总车厢数']] = self.data[['识别正确数', '参与统计数', '总车厢数']].fillna(0)
                
            except Exception as e:
                raise ValueError(f"数据类型转换失败：{e}")
            
            # 数据验证
            if len(self.data) == 0:
                raise ValueError("数据文件为空")
            
            # 打印数据信息用于调试
            print(f"\n数据加载成功：")
            print(f"- 总记录数：{len(self.data)}")
            print(f"- 日期范围：{self.data['日期'].min()} 至 {self.data['日期'].max()}")
            print(f"- 示例数据：")
            print(self.data[['列车名称', '日期时间', '识别正确数', '参与统计数']].head(3).to_string())
            
            self.update_basic_info()
            self.analyze_btn.setEnabled(True)
            
            self.status_bar.showMessage(f"数据加载成功！共 {len(self.data)} 条记录")
            QMessageBox.information(self, "成功", f"数据加载成功！\n共加载 {len(self.data)} 条记录")
            
        except Exception as e:
            self.status_bar.showMessage("数据加载失败")
            QMessageBox.critical(self, "错误", f"数据加载失败：{str(e)}")
    
    def update_basic_info(self):
        """更新基本信息显示"""
        total_accuracy = (self.data['识别正确数'].sum() / self.data['参与统计数'].sum() * 100)
        
        info = f"""📊 数据概况
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 总记录数：{len(self.data):,} 条
• 数据时间范围：{self.data['日期'].min()} 至 {self.data['日期'].max()}
• 总车厢数：{self.data['总车厢数'].sum():,} 节
• 总识别正确数：{self.data['识别正确数'].sum():,} 节
• 总参与统计数：{self.data['参与统计数'].sum():,} 节

📈 整体统计
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 总体准确率：{total_accuracy:.2f}%
• 平均单车准确率：{self.data['准确率(%)'].mean():.2f}%
• 最高单车准确率：{self.data['准确率(%)'].max():.2f}%
• 最低单车准确率：{self.data['准确率(%)'].min():.2f}%

🎯 使用说明
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 选择开始和结束日期
2. 点击"生成分析"按钮进行数据分析
3. 点击"显示图表"按钮查看可视化图表

请选择时间范围后点击"生成分析"。
"""
        
        self.info_text.setPlainText(info)
    
    def start_analysis(self):
        """开始数据分析"""
        if self.data is None:
            QMessageBox.warning(self, "警告", "请先加载数据！")
            return
        
        start_date = self.start_date_edit.date().toPython()
        end_date = self.end_date_edit.date().toPython()
        
        if start_date > end_date:
            QMessageBox.warning(self, "警告", "开始日期不能晚于结束日期！")
            return
        
        try:
            filtered_data = self.data[
                (self.data['日期'] >= start_date) & 
                (self.data['日期'] <= end_date)
            ].copy()
            
            if len(filtered_data) == 0:
                QMessageBox.warning(self, "警告", "选择的时间范围内没有数据！")
                return
            
            # 计算统计信息
            total_correct = filtered_data['识别正确数'].sum()
            total_participated = filtered_data['参与统计数'].sum()
            period_accuracy = (total_correct / total_participated * 100) if total_participated > 0 else 0
            
            # 按日统计
            daily_stats = filtered_data.groupby('日期').agg({
                '识别正确数': 'sum',
                '参与统计数': 'sum',
                '总车厢数': 'sum',
                '列车名称': 'count'
            }).reset_index()
            
            daily_stats['日准确率'] = (daily_stats['识别正确数'] / daily_stats['参与统计数'] * 100).round(2)
            
            self.current_stats = {
                'period_accuracy': period_accuracy,
                'total_correct': total_correct,
                'total_participated': total_participated,
                'total_trains': len(filtered_data),
                'daily_stats': daily_stats,
                'filtered_data': filtered_data,
                'start_date': start_date,
                'end_date': end_date
            }
            
            # 更新信息显示
            self.update_period_info(self.current_stats)
            
            # 恢复界面
            self.chart_btn.setEnabled(True)
            self.status_bar.showMessage("分析完成")
            
            QMessageBox.information(self, "完成", "数据分析完成！点击'显示图表'查看可视化结果。")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"分析失败：{str(e)}")
    
    def update_period_info(self, stats):
        """更新时间段统计信息"""
        daily_stats = stats['daily_stats']
        filtered_data = stats['filtered_data']
        
        # 计算等级分布
        excellent = len(filtered_data[filtered_data['准确率(%)'] >= 95])
        good = len(filtered_data[(filtered_data['准确率(%)'] >= 90) & (filtered_data['准确率(%)'] < 95)])
        fair = len(filtered_data[(filtered_data['准确率(%)'] >= 80) & (filtered_data['准确率(%)'] < 90)])
        poor = len(filtered_data[filtered_data['准确率(%)'] < 80])
        
        info = f"""🚂 时间范围分析报告
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 分析时间段：{stats['start_date']} 至 {stats['end_date']}

🎯 核心指标
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 时间段总体准确率：{stats['period_accuracy']:.2f}%
• 总识别正确数：{stats['total_correct']:,} 节
• 总参与统计数：{stats['total_participated']:,} 节
• 列车总数：{stats['total_trains']:,} 趟

📊 日均统计
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 统计天数：{len(daily_stats)} 天
• 日均列车数：{stats['total_trains'] / len(daily_stats):.1f} 趟
• 日均车厢数：{stats['total_participated'] / len(daily_stats):.1f} 节
• 日均准确率：{daily_stats['日准确率'].mean():.2f}%

📈 准确率分布
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 最高日准确率：{daily_stats['日准确率'].max():.2f}%
• 最低日准确率：{daily_stats['日准确率'].min():.2f}%
• 平均单车准确率：{filtered_data['准确率(%)'].mean():.2f}%
• 准确率标准差：{daily_stats['日准确率'].std():.2f}%

🏆 准确率等级分布
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🟢 优秀 (≥95%)：{excellent} 趟 ({excellent/len(filtered_data)*100:.1f}%)
🟡 良好 (90-95%)：{good} 趟 ({good/len(filtered_data)*100:.1f}%)
🟠 一般 (80-90%)：{fair} 趟 ({fair/len(filtered_data)*100:.1f}%)
🔴 待改进 (<80%)：{poor} 趟 ({poor/len(filtered_data)*100:.1f}%)

📅 详细日统计
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        for _, row in daily_stats.iterrows():
            info += f"\n📅 {row['日期']}：{row['列车名称']}趟车，{row['参与统计数']}节车厢，准确率{row['日准确率']}%"
        
        self.info_text.setPlainText(info)
    
    def show_charts(self):
        """显示图表"""
        if self.current_stats is None:
            QMessageBox.warning(self, "警告", "请先生成分析报告！")
            return
        
        try:
            self.create_charts(self.current_stats)
            self.status_bar.showMessage("图表显示完成")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"图表生成失败：{str(e)}")
    
    def create_charts(self, stats):
        """创建图表"""
        # 重新设置字体以确保显示正确
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 10
        
        # 设置图表样式
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('default')
        
        # 创建大窗口
        fig = plt.figure(figsize=(16, 10))
        
        # 设置整体字体
        for text in fig.findobj(plt.Text):
            text.set_fontfamily(['Microsoft YaHei', 'SimHei'])
        
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'🚂 列车积水识别准确率综合分析报告\n📅 {stats["start_date"]} 至 {stats["end_date"]}', 
                    fontsize=16, fontweight='bold', fontfamily=['Microsoft YaHei', 'SimHei'])
        
        daily_stats = stats['daily_stats']
        filtered_data = stats['filtered_data']
        
        # 1. 日准确率趋势图 (大图)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(daily_stats['日期'], daily_stats['日准确率'], 
                marker='o', linewidth=3, markersize=8, color='#2E86C1')
        ax1.fill_between(daily_stats['日期'], daily_stats['日准确率'], 
                        alpha=0.3, color='#2E86C1')
        
        mean_accuracy = daily_stats['日准确率'].mean()
        ax1.axhline(y=mean_accuracy, color='red', linestyle='--', 
                   label=f'平均值: {mean_accuracy:.2f}%')
        ax1.axhline(y=95, color='green', linestyle=':', label='优秀线: 95%')
        ax1.axhline(y=90, color='orange', linestyle=':', label='良好线: 90%')
        
        ax1.set_title('📈 日准确率趋势分析', fontsize=14, fontweight='bold', fontfamily=['Microsoft YaHei', 'SimHei'])
        ax1.set_ylabel('准确率 (%)', fontsize=12, fontfamily=['Microsoft YaHei', 'SimHei'])
        ax1.set_xlabel('日期', fontsize=12, fontfamily=['Microsoft YaHei', 'SimHei'])
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. 日车厢数量统计
        ax2 = fig.add_subplot(gs[1, 0])
        bars = ax2.bar(daily_stats['日期'], daily_stats['参与统计数'], 
                      color='#28B463', alpha=0.7)
        ax2.set_title('📊 日车厢数量', fontsize=12, fontweight='bold', fontfamily=['Microsoft YaHei', 'SimHei'])
        ax2.set_ylabel('车厢数', fontfamily=['Microsoft YaHei', 'SimHei'])
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        
        # 3. 日列车数量统计
        ax3 = fig.add_subplot(gs[1, 1])
        bars3 = ax3.bar(daily_stats['日期'], daily_stats['列车名称'], 
                       color='#8E44AD', alpha=0.7)
        ax3.set_title('🚂 日列车数量', fontsize=12, fontweight='bold', fontfamily=['Microsoft YaHei', 'SimHei'])
        ax3.set_ylabel('列车数', fontfamily=['Microsoft YaHei', 'SimHei'])
        ax3.tick_params(axis='x', rotation=45, labelsize=8)
        
        # 4. 准确率等级分布饼图
        ax4 = fig.add_subplot(gs[1, 2])
        excellent = len(filtered_data[filtered_data['准确率(%)'] >= 95])
        good = len(filtered_data[(filtered_data['准确率(%)'] >= 90) & (filtered_data['准确率(%)'] < 95)])
        fair = len(filtered_data[(filtered_data['准确率(%)'] >= 80) & (filtered_data['准确率(%)'] < 90)])
        poor = len(filtered_data[filtered_data['准确率(%)'] < 80])
        
        sizes = [excellent, good, fair, poor]
        labels = ['优秀≥95%', '良好90-95%', '一般80-90%', '待改进<80%']
        colors = ['#2ECC71', '#F39C12', '#E67E22', '#E74C3C']
        
        # 只显示非零部分
        non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
        if non_zero:
            sizes_nz, labels_nz, colors_nz = zip(*non_zero)
            wedges, texts, autotexts = ax4.pie(sizes_nz, labels=labels_nz, colors=colors_nz, 
                                              autopct='%1.1f%%', startangle=90,
                                              textprops={'fontfamily': ['Microsoft YaHei', 'SimHei']})
            ax4.set_title('🎯 准确率分布', fontsize=12, fontweight='bold', fontfamily=['Microsoft YaHei', 'SimHei'])
            
            # 设置饼图文本字体
            for text in texts:
                text.set_fontfamily(['Microsoft YaHei', 'SimHei'])
            for autotext in autotexts:
                autotext.set_fontfamily(['Microsoft YaHei', 'SimHei'])
        
        # 设置图表所有元素的字体
        def set_chinese_font(ax):
            """为坐标轴设置中文字体"""
            ax.title.set_fontfamily(['Microsoft YaHei', 'SimHei'])
            ax.xaxis.label.set_fontfamily(['Microsoft YaHei', 'SimHei'])
            ax.yaxis.label.set_fontfamily(['Microsoft YaHei', 'SimHei'])
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily(['Microsoft YaHei', 'SimHei'])
            if ax.legend_:
                for text in ax.legend_.get_texts():
                    text.set_fontfamily(['Microsoft YaHei', 'SimHei'])
        
        # 应用字体设置到所有坐标轴
        set_chinese_font(ax1)
        set_chinese_font(ax2)
        set_chinese_font(ax3)
        set_chinese_font(ax4)
        
        plt.tight_layout()
        plt.show()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 设置现代化样式
    
    window = TrainAccuracyMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()