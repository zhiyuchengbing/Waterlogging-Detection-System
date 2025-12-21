import threading
import time
import os
from datetime import datetime
from test_mysql import load_database_config, create_database_tables, insert_train_data, insert_carriage_data
from test_mysql import find_image_for_carriage_with_csv_cached, find_side_image_by_plate_number_cached
import csv

class DatabaseAutoUpdater:
    """
    数据库自动更新器类
    监控新数据并自动添加到数据库中
    """
    
    def __init__(self, check_interval=30):
        """
        初始化数据库自动更新器
        
        Args:
            check_interval (int): 检查间隔（秒），默认30秒
        """
        self.check_interval = check_interval
        self.running = False
        self.monitor_thread = None
        self.train_dir = "F:\\baowen"  # 列车信息目录
        self.output_dir = "runs/output_images"  # 输出目录
        self.stop_event = threading.Event()
        
        # 确保数据库表已创建
        self._ensure_database_tables()
    
    def _ensure_database_tables(self):
        """确保数据库表已创建"""
        try:
            create_database_tables()
            print("数据库表结构已确认存在")
        except Exception as e:
            print(f"创建数据库表时出错: {e}")
    
    def start_monitoring(self):
        """
        启动监控线程
        """
        if self.running:
            print("监控已在运行中")
            return
            
        self.running = True
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"数据库自动更新监控已启动，检查间隔: {self.check_interval}秒")
    
    def stop_monitoring(self):
        """
        停止监控线程
        """
        if not self.running:
            print("监控未在运行中")
            return
            
        self.running = False
        self.stop_event.set()  # 立即唤醒等待
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("数据库自动更新监控已停止")
    
    def _monitor_loop(self):
        """
        监控循环
        """
        while self.running:
            try:
                self._check_and_update_data()
                # 使用事件等待，可被stop唤醒，避免长时间阻塞关闭
                if self.stop_event.wait(self.check_interval):
                    break
            except Exception as e:
                print(f"监控过程中出错: {e}")
                if self.stop_event.wait(self.check_interval):
                    break
    
    def _check_and_update_data(self):
        """
        检查并更新数据
        """
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 检查新数据...")
        
        # 检查列车信息目录
        if not os.path.exists(self.train_dir):
            print(f"列车信息目录不存在: {self.train_dir}")
            return
        
        # 获取所有txt文件
        txt_files = [
            f for f in os.listdir(self.train_dir)
            if f.lower().endswith('.txt') and os.path.isfile(os.path.join(self.train_dir, f))
        ]
        
        if not txt_files:
            print("未找到列车信息文件")
            return

        # 只处理最近一周内的列车文件，避免对很久以前的数据反复扫描
        one_week_ago = datetime.now().timestamp() - 7 * 24 * 60 * 60

        # 处理每个列车文件
        for txt_file in txt_files:
            try:
                train_name = os.path.splitext(txt_file)[0]  # 去掉.txt后缀

                # 列车名通常为时间戳，如 20250702113405，仅当能解析且在最近一周内时才处理
                try:
                    dt = datetime.strptime(train_name, "%Y%m%d%H%M%S")
                    if dt.timestamp() < one_week_ago:
                        # 太早的数据直接跳过
                        continue
                except ValueError:
                    # 如果列车名不是时间格式，则按原逻辑处理
                    pass

                self._process_train_file(train_name, txt_file)
            except Exception as e:
                print(f"处理列车文件 {txt_file} 时出错: {e}")
                continue
    
    def _process_train_file(self, train_name, txt_file):
        """
        处理单个列车文件
        
        Args:
            train_name (str): 列车名称
            txt_file (str): txt文件名
        """
        file_path = os.path.join(self.train_dir, txt_file)
        
        if not os.path.exists(file_path):
            return
            
        # 读取列车信息
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            
            # 计算列车统计数据
            total_carriages = 0
            correct_count = 0
            
            # 分别存储普通车号和特殊车号
            normal_carriages = []
            special_carriages = []
            
            for row in reader:
                if len(row) >= 5:  # 确保至少有序号、型号、车号、时间、备注
                    carriage_id = row[0]  # 序号，如"01"
                    carriage_type = row[1].strip()  # 型号，如"C70E"
                    carriage_number = row[2].strip()  # 车号，如"1690345"
                    carriage_time = row[3].strip()  # 时间
                    
                    # 完整车号 = 型号 + 车号
                    full_carriage_number = f"{carriage_type}{carriage_number}"
                    
                    # 判断是否为特殊车号（P或N开头）
                    is_special = full_carriage_number.upper().startswith(('P', 'N'))
                    
                    # 查找对应顶部图片和积水信息（直接从CSV读取）
                    top_image_info = find_image_for_carriage_with_csv_cached(train_name, carriage_id, self.output_dir)
                    
                    # 查找对应侧面图片和车号识别结果
                    side_image_info = find_side_image_by_plate_number_cached(train_name, full_carriage_number)
                    
                    recognized_plate = "未识别"
                    recognized_time = "未知"
                    
                    if side_image_info:
                        recognized_plate = side_image_info['plate_number']
                        recognized_time = side_image_info['timestamp']
                    
                    # 正确设置积水信息：从top_image_info中获取是否有水的信息
                    water_info = '有水' if top_image_info.get('has_water', False) else '无水'
                    water_area_ratio = top_image_info.get('water_area_ratio', 0.00)
                    
                    # 创建车厢信息字典
                    carriage_info = {
                        'carriage_id': carriage_id,
                        'full_carriage_number': full_carriage_number,
                        'recognized_plate': recognized_plate,
                        'recognized_time': recognized_time,
                        'top_image_info': top_image_info,
                        'side_image_info': side_image_info,
                        'is_special': is_special,
                        'water_info': water_info,
                        'water_area_ratio': water_area_ratio
                    }
                    
                    # 根据是否为特殊车号分类存储
                    if is_special:
                        special_carriages.append(carriage_info)
                    else:
                        normal_carriages.append(carriage_info)
                        
                    # 统计信息
                    total_carriages += 1
                    if water_info == '无水':
                        correct_count += 1
            
            # 计算准确率
            accuracy_rate = (correct_count / total_carriages * 100) if total_carriages > 0 else 0.0
            
            # 插入列车数据
            insert_train_data(train_name, total_carriages, accuracy_rate, correct_count, total_carriages)
            
            # 先添加普通车号，再添加特殊车号（与界面显示顺序保持一致）
            all_carriages = normal_carriages + special_carriages
            
            # 处理每个车厢数据并插入数据库
            for carriage_info in all_carriages:
                top_image_info = carriage_info['top_image_info']
                side_image_info = carriage_info['side_image_info']
                
                # 创建车厢数据字典
                carriage_data = {
                    'carriage_id': carriage_info['carriage_id'],
                    'full_carriage_number': carriage_info['full_carriage_number'],
                    'recognized_plate': carriage_info['recognized_plate'],
                    'recognition_time': carriage_info['recognized_time'],
                    'is_special': carriage_info['is_special'],
                    'water_info': carriage_info['water_info'],
                    'water_area_ratio': carriage_info['water_area_ratio'],
                    'top_image_path': top_image_info.get('water_detected_path') or top_image_info.get('original_path', ''),
                    'side_image_path': side_image_info['image_path'] if side_image_info and side_image_info.get('image_path') else ''
                }
                
                # 插入车厢数据
                insert_carriage_data(train_name, carriage_data)
            
            print(f"已处理列车: {train_name}，共{total_carriages}节车厢")

    def update_single_train(self, train_name):
        """
        更新单个列车数据
        
        Args:
            train_name (str): 列车名称
        """
        txt_file = f"{train_name}.txt"
        file_path = os.path.join(self.train_dir, txt_file)
        
        if not os.path.exists(file_path):
            print(f"列车文件不存在: {file_path}")
            return
            
        self._process_train_file(train_name, txt_file)