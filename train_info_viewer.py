import sys
import os
import pymysql
import yaml
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QComboBox, QTableWidget, QTableWidgetItem, QLabel, QPushButton,
                               QScrollArea, QHeaderView, QSplitter, QFrame)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QColor

class DatabaseConfig:
    """数据库配置管理类"""
    
    @staticmethod
    def load_config():
        """从YAML文件加载数据库配置"""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'settings.yaml')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config['database']
        except FileNotFoundError:
            print(f"配置文件未找到: {config_path}")
            return None
        except Exception as e:
            print(f"读取配置文件时出错: {e}")
            return None

class DatabaseManager:
    """数据库管理类"""
    
    def __init__(self):
        self.db_config = DatabaseConfig.load_config()
    
    def get_connection(self):
        """获取数据库连接"""
        if not self.db_config:
            raise Exception("无法加载数据库配置")
            
        return pymysql.connect(
            host=self.db_config['host'],
            port=int(self.db_config['port']),
            user=self.db_config['username'],
            password=str(self.db_config['password']),
            database=self.db_config['database_name'],
            charset=self.db_config['charset'],
            cursorclass=pymysql.cursors.DictCursor
        )
    
    def get_all_trains(self):
        """获取所有列车信息"""
        connection = None
        try:
            connection = self.get_connection()
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT id, train_name, total_carriages, accuracy_rate, 
                           correct_count, participated_count, created_at, updated_at
                    FROM trains 
                    ORDER BY updated_at DESC
                """)
                results = cursor.fetchall()
                return results
        except Exception as e:
            print(f"获取列车信息失败: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    def get_train_carriages(self, train_id):
        """获取指定列车的车厢信息"""
        connection = None
        try:
            connection = self.get_connection()
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT id, carriage_id, full_carriage_number, recognized_plate,
                           recognition_time, is_special, water_info, water_area_ratio,
                           top_image_path, side_image_path, created_at, updated_at
                    FROM carriages 
                    WHERE train_id = %s 
                    ORDER BY CAST(carriage_id AS UNSIGNED)
                """, (train_id,))
                results = cursor.fetchall()
                return results
        except Exception as e:
            print(f"获取车厢信息失败: {e}")
            return []
        finally:
            if connection:
                connection.close()

class TrainInfoViewer(QMainWindow):
    """列车信息查看窗口"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.db_manager = DatabaseManager()
        self.current_train_id = None
        self.init_ui()
        self.load_train_list()
    
    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle("列车信息查看")
        self.setGeometry(100, 100, 1400, 900)
        
        # 设置应用样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fa;
            }
            QLabel {
                color: #2c3e50;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1c6ea4;
            }
            QTableWidget {
                border: 1px solid #e1e1e1;
                border-radius: 6px;
                background-color: white;
                gridline-color: #f0f0f0;
            }
            QTableWidget::item {
                padding: 5px;
                border-bottom: 1px solid #eeeeee;
            }
            QTableWidget::item:selected {
                background-color: #e8f0fe;
                color: #2c3e50;
            }
            QHeaderView::section {
                background-color: #3498db;
                color: white;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
        """)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 左侧区域 - 列车选择和车厢列表
        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")
        left_panel.setStyleSheet("""
            #leftPanel {
                background-color: white;
                border-radius: 10px;
                border: 1px solid #e1e1e1;
            }
        """)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)
        
        # 列车选择区域
        train_select_layout = QHBoxLayout()
        
        train_select_layout.addWidget(QLabel("选择列车:"))
        self.train_combo = QComboBox()
        self.train_combo.setMinimumWidth(250)
        self.train_combo.currentIndexChanged.connect(self.on_train_selected)
        self.train_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
            }
            QComboBox::drop-down {
                border: 0px;
                width: 20px;
            }
        """)
        train_select_layout.addWidget(self.train_combo)
        
        # 上一趟按钮
        self.prev_train_btn = QPushButton("← 上一趟")
        self.prev_train_btn.setStyleSheet("background-color: #2ecc71;")
        self.prev_train_btn.clicked.connect(self.go_to_previous_train)
        train_select_layout.addWidget(self.prev_train_btn)
        
        # 下一趟按钮
        self.next_train_btn = QPushButton("下一趟 →")
        self.next_train_btn.setStyleSheet("background-color: #e74c3c;")
        self.next_train_btn.clicked.connect(self.go_to_next_train)
        train_select_layout.addWidget(self.next_train_btn)
        
        left_layout.addLayout(train_select_layout)
        
        # 车厢信息表格
        self.carriage_table = QTableWidget()
        self.carriage_table.setColumnCount(6)
        self.carriage_table.setHorizontalHeaderLabels([
            "序号", "车号", "识别车号", "识别时间", "积水信息", "积水面积占比(%)"
        ])
        # 修改列宽设置，给识别时间列分配更多空间
        self.carriage_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.carriage_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.carriage_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        # 为识别时间列设置固定宽度，确保能完整显示时间信息
        self.carriage_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.carriage_table.setColumnWidth(3, 150)  # 设置识别时间列宽度为150像素
        self.carriage_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.carriage_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.carriage_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.carriage_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.carriage_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.carriage_table.setAlternatingRowColors(True)
        self.carriage_table.currentCellChanged.connect(self.on_table_row_selected)
        
        # 设置黑白配色三线表样式
        self.carriage_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid black;
                gridline-color: white;
                background-color: white;
                alternate-background-color: #f8f8f8;
            }
            QTableWidget::item {
                border-bottom: 1px solid #e0e0e0;
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #e0e0e0;
                color: black;
            }
            QHeaderView::section {
                background-color: white;
                color: black;
                font-weight: bold;
                border: none;
                border-bottom: 2px solid black;
                border-top: 2px solid black;
                padding: 6px;
            }
            QHeaderView::section:first {
                border-left: none;
            }
            QHeaderView::section:last {
                border-right: none;
            }
            QTableCornerButton::section {
                background-color: white;
                border: none;
            }
        """)
        
        left_layout.addWidget(self.carriage_table, 1)
        
        # 右侧图片显示区域
        right_panel = QWidget()
        right_panel.setObjectName("rightPanel")
        right_panel.setStyleSheet("""
            #rightPanel {
                background-color: white;
                border-radius: 10px;
                border: 1px solid #e1e1e1;
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 15, 15, 15)
        
        # 添加准确率显示区域到右上角
        accuracy_container = QWidget()
        accuracy_container.setObjectName("accuracyContainer")
        accuracy_container.setStyleSheet("""
            #accuracyContainer {
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #e9ecef;
                padding: 10px;
            }
        """)
        accuracy_layout = QHBoxLayout(accuracy_container)
        accuracy_layout.setContentsMargins(10, 10, 10, 10)
        
        # 准确率标签
        accuracy_label = QLabel("识别准确率:")
        accuracy_label.setStyleSheet("font-weight: bold; color: #495057; font-size: 14px;")
        accuracy_layout.addWidget(accuracy_label)
        
        # 圆形准确率显示
        self.accuracy_circle = QLabel()
        self.accuracy_circle.setFixedSize(80, 80)
        self.accuracy_circle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.accuracy_circle.setStyleSheet("""
            QLabel {
                background-color: qradialgradient(
                    cx: 0.5, cy: 0.5, radius: 0.5,
                    stop: 0 #ffffff,
                    stop: 0.8 #e3f2fd,
                    stop: 1 #2196f3
                );
                border: 3px solid #1976d2;
                border-radius: 40px;
                color: #1565c0;
                font-weight: bold;
                font-size: 16px;
            }
        """)
        self.accuracy_circle.setText("0%")
        accuracy_layout.addWidget(self.accuracy_circle)
        
        # 添加当前页准确率信息标签
        self.page_accuracy_info = QLabel("当前页准确率: 0%")
        self.page_accuracy_info.setStyleSheet("font-weight: bold; color: #495057; font-size: 12px;")
        accuracy_layout.addWidget(self.page_accuracy_info)
        
        # 添加弹性空间
        accuracy_layout.addStretch()
        
        right_layout.addWidget(accuracy_container)
        
        # 图片显示区域
        images_layout = QVBoxLayout()
        
        # 顶部图片区域
        self.image_scroll_area = QScrollArea()
        self.image_scroll_area.setWidgetResizable(True)
        self.image_scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.image_scroll_area.setMinimumHeight(350)
        self.image_scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
        """)
        
        self.image_label = QLabel("无图片")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            background-color: white;
            color: #7f8c8d;
            font-size: 16px;
            border-radius: 5px;
        """)
        
        self.image_scroll_area.setWidget(self.image_label)
        images_layout.addWidget(self.image_scroll_area)
        
        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: black;")
        images_layout.addWidget(separator)
        
        # 侧面图片区域
        self.side_image_scroll_area = QScrollArea()
        self.side_image_scroll_area.setWidgetResizable(True)
        self.side_image_scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.side_image_scroll_area.setMinimumHeight(350)
        self.side_image_scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
        """)
        
        self.side_image_label = QLabel("无侧面图片")
        self.side_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.side_image_label.setStyleSheet("""
            background-color: white;
            color: #7f8c8d;
            font-size: 16px;
            border-radius: 5px;
        """)
        
        self.side_image_scroll_area.setWidget(self.side_image_label)
        images_layout.addWidget(self.side_image_scroll_area)
        
        right_layout.addLayout(images_layout)
        
        # 设置左右面板比例
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([2, 3])
        
        main_layout.addWidget(splitter)
        
        # 居中显示
        self.center_window()
    
    def center_window(self):
        """使窗口居中显示在桌面上"""
        # 获取屏幕几何信息
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        
        # 计算窗口位置
        window_width = self.width()
        window_height = self.height()
        x = (screen_geometry.width() - window_width) // 2
        y = (screen_geometry.height() - window_height) // 2
        
        # 设置窗口位置
        self.move(x, y)
    
    def load_train_list(self):
        """加载列车列表"""
        try:
            trains = self.db_manager.get_all_trains()
            self.train_combo.clear()
            
            # 按列车名称（时间格式）倒序排序，最新的在前
            # train_name格式如：20250915071121
            sorted_trains = sorted(trains, key=lambda x: x['train_name'], reverse=True)
            
            for train in sorted_trains:
                display_text = f"{train['train_name']} (准确率: {train['accuracy_rate']:.2f}%)"
                self.train_combo.addItem(display_text, train['id'])
                
            # 更新导航按钮状态
            self.update_navigation_buttons()
                
        except Exception as e:
            print(f"加载列车列表失败: {e}")
    
    def on_train_selected(self, index):
        """当选择列车时"""
        if index < 0:
            return
            
        train_id = self.train_combo.currentData()
        self.current_train_id = train_id
        self.load_carriage_data(train_id)
        
        # 更新导航按钮状态
        self.update_navigation_buttons()
    
    def load_carriage_data(self, train_id):
        """加载车厢数据"""
        try:
            # 获取列车信息以显示准确率
            trains = self.db_manager.get_all_trains()
            accuracy_rate = 0.0
            for train in trains:
                if train['id'] == train_id:
                    accuracy_rate = float(train['accuracy_rate'])
                    self.update_accuracy_display(accuracy_rate)
                    break
            
            # 获取车厢信息
            carriages = self.db_manager.get_train_carriages(train_id)
            
            # 清空表格
            self.carriage_table.setRowCount(0)
            
            # 添加到表格
            for carriage in carriages:
                row_position = self.carriage_table.rowCount()
                self.carriage_table.insertRow(row_position)
                
                # 设置表格项
                item0 = QTableWidgetItem(carriage['carriage_id'])
                self.carriage_table.setItem(row_position, 0, item0)
                self.carriage_table.setItem(row_position, 1, QTableWidgetItem(carriage['full_carriage_number']))
                self.carriage_table.setItem(row_position, 2, QTableWidgetItem(carriage['recognized_plate']))
                self.carriage_table.setItem(row_position, 3, QTableWidgetItem(carriage['recognition_time']))
                self.carriage_table.setItem(row_position, 4, QTableWidgetItem(carriage['water_info']))
                
                # 设置积水面积占比
                water_ratio_text = f"{carriage['water_area_ratio']:.2f}%" if carriage['water_area_ratio'] > 0 else "0.0%"
                self.carriage_table.setItem(row_position, 5, QTableWidgetItem(water_ratio_text))
                
                # 存储图片路径数据
                if item0:  # 检查item0是否为None
                    item0.setData(Qt.ItemDataRole.UserRole, {
                        'top_image_path': carriage['top_image_path'],
                        'side_image_path': carriage['side_image_path']
                    })
                
                # 为特殊车号设置不同的背景色
                if carriage['is_special']:
                    for col in range(self.carriage_table.columnCount()):
                        item = self.carriage_table.item(row_position, col)
                        if item:
                            item.setBackground(QColor(255, 255, 255))
            
            # 如果有数据，选择第一行
            if self.carriage_table.rowCount() > 0:
                self.carriage_table.selectRow(0)
                
        except Exception as e:
            print(f"加载车厢数据失败: {e}")
    
    def update_accuracy_display(self, accuracy_rate):
        """更新准确率显示"""
        # 更新准确率显示
        self.accuracy_circle.setText(f"{accuracy_rate:.1f}%")
        
        # 只显示简单的准确率信息
        self.page_accuracy_info.setText(f"当前页准确率: {accuracy_rate:.1f}%")
        
        # 根据准确率设置不同的颜色
        if accuracy_rate >= 80:
            # 高准确率：白色背景 + 绿色边框和文字
            self.accuracy_circle.setStyleSheet("""
                QLabel {
                    background-color: #ffffff;
                    border: 2px solid #28a745;
                    border-radius: 40px;
                    color: #28a745;
                    font-weight: 600;
                    font-size: 16px;
                }
            """)
        elif accuracy_rate >= 60:
            # 中等准确率：白色背景 + 橙色边框和文字
            self.accuracy_circle.setStyleSheet("""
                QLabel {
                    background-color: #ffffff;
                    border: 2px solid #ff8c00;
                    border-radius: 40px;
                    color: #ff8c00;
                    font-weight: 600;
                    font-size: 16px;
                }
            """)
        else:
            # 低准确率：白色背景 + 红色边框和文字
            self.accuracy_circle.setStyleSheet("""
                QLabel {
                    background-color: #ffffff;
                    border: 2px solid #dc3545;
                    border-radius: 40px;
                    color: #dc3545;
                    font-weight: 600;
                    font-size: 16px;
                }
            """)
    
    def on_table_row_selected(self, current_row, current_column, previous_row, previous_column):
        """当表格行被选中时"""
        if current_row < 0:
            return
            
        # 获取图片路径信息
        item = self.carriage_table.item(current_row, 0)
        if item:  # 检查item是否为None
            image_info = item.data(Qt.ItemDataRole.UserRole)
            
            if image_info:
                top_image_path = image_info.get('top_image_path', '')
                side_image_path = image_info.get('side_image_path', '')
                
                # 显示顶部图片
                if top_image_path and os.path.exists(top_image_path):
                    self.display_image(self.image_scroll_area, self.image_label, top_image_path)
                else:
                    # 检查image_label是否仍然有效
                    try:
                        self.image_label.setText("无图片")
                    except RuntimeError:
                        # 如果对象已被删除，创建新的标签
                        self.image_label = QLabel("无图片")
                        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                        self.image_label.setStyleSheet("""
                            background-color: white;
                            color: #7f8c8d;
                            font-size: 16px;
                            border-radius: 5px;
                        """)
                        self.image_scroll_area.setWidget(self.image_label)
                
                # 显示侧面图片
                if side_image_path and os.path.exists(side_image_path):
                    self.display_image(self.side_image_scroll_area, self.side_image_label, side_image_path)
                else:
                    # 检查side_image_label是否仍然有效
                    try:
                        self.side_image_label.setText("无侧面图片")
                    except RuntimeError:
                        # 如果对象已被删除，创建新的标签
                        self.side_image_label = QLabel("无侧面图片")
                        self.side_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                        self.side_image_label.setStyleSheet("""
                            background-color: white;
                            color: #7f8c8d;
                            font-size: 16px;
                            border-radius: 5px;
                        """)
                        self.side_image_scroll_area.setWidget(self.side_image_label)
    
    def display_image(self, scroll_area, label, image_path):
        """显示图片"""
        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # 创建新的QLabel以适应图片大小
                image_label = QLabel()
                image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                
                # 调整图片大小，适应窗口但不超过原始大小的90%
                max_width = scroll_area.width() * 0.9
                max_height = scroll_area.height() * 0.9
                
                # 如果图片比设定的最大尺寸大，则缩小
                if pixmap.width() > max_width or pixmap.height() > max_height:
                    pixmap = pixmap.scaled(
                        int(max_width),
                        int(max_height),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                
                image_label.setPixmap(pixmap)
                scroll_area.setWidget(image_label)
                
                # 更新标签引用
                if scroll_area == self.image_scroll_area:
                    self.image_label = image_label
                elif scroll_area == self.side_image_scroll_area:
                    self.side_image_label = image_label
            else:
                label.setText("图片加载失败")
        except Exception as e:
            label.setText(f"图片加载错误: {str(e)}")
    
    def update_navigation_buttons(self):
        """更新导航按钮的启用状态"""
        current_index = self.train_combo.currentIndex()
        total_count = self.train_combo.count()
        
        # 更新上一趟按钮状态
        if current_index <= 0 or total_count <= 1:
            self.prev_train_btn.setEnabled(False)
            self.prev_train_btn.setStyleSheet("background-color: #95a5a6; color: #7f8c8d;")
        else:
            self.prev_train_btn.setEnabled(True)
            self.prev_train_btn.setStyleSheet("background-color: #2ecc71;")
        
        # 更新下一趟按钮状态
        if current_index >= total_count - 1 or total_count <= 1:
            self.next_train_btn.setEnabled(False)
            self.next_train_btn.setStyleSheet("background-color: #95a5a6; color: #7f8c8d;")
        else:
            self.next_train_btn.setEnabled(True)
            self.next_train_btn.setStyleSheet("background-color: #e74c3c;")
    
    def go_to_previous_train(self):
        """切换到上一趟列车"""
        current_index = self.train_combo.currentIndex()
        if current_index > 0:
            self.train_combo.setCurrentIndex(current_index - 1)
    
    def go_to_next_train(self):
        """切换到下一趟列车"""
        current_index = self.train_combo.currentIndex()
        if current_index < self.train_combo.count() - 1:
            self.train_combo.setCurrentIndex(current_index + 1)

def main():
    """主函数"""
    app = QApplication(sys.argv)
    window = TrainInfoViewer()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()