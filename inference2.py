
import os
import cv2
import numpy as np
import csv
import json
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from datetime import datetime
import time
import hashlib
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 固定积水识别根目录和模型权重为绝对路径，保持与主程序一致
RUNS_OUTPUT_IMAGES_DIR = r"E:\积水识别项目\demo0625\demo\runs\output_images"
WATER_SEGMENT_WEIGHTS = r"E:\积水识别项目\demo0625\demo\runs\segment\train\weights_jishui\best4.pt"
PROCESSED_FILES_PATH = r"E:\积水识别项目\demo0625\demo\已处理文件记录.txt"

class WaterDetectionMonitor(FileSystemEventHandler):
    """
    文件夹监控器，自动检测新文件并进行积水识别
    """
    
    def __init__(self, model_path, root_folder, processed_files_file, check_interval=5):
        self.model_path = model_path
        self.root_folder = Path(root_folder)
        self.processed_files_file = Path(processed_files_file)
        self.check_interval = check_interval
        self.model = None
        self.processed_files = set()
        self.processing_lock = threading.Lock()
        
        # 加载模型
        self._load_model()
        
        # 加载已处理文件记录
        self._load_processed_files()
        
        # 支持的图片格式
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        print(f"监控器已启动，监控目录: {self.root_folder}")
        print(f"检查间隔: {self.check_interval} 秒")
    
    def _load_model(self):
        """加载YOLO模型"""
        try:
            print("正在加载模型...")
            self.model = YOLO(self.model_path)
            print("模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.model = None
    
    def _load_processed_files(self):
        """加载已处理文件记录"""
        if self.processed_files_file.exists():
            try:
                with open(self.processed_files_file, 'r', encoding='utf-8') as f:
                    self.processed_files = set(line.strip() for line in f if line.strip())
                print(f"已加载 {len(self.processed_files)} 个已处理文件记录")
            except Exception as e:
                print(f"加载已处理文件记录失败: {e}")
                self.processed_files = set()
        else:
            print("未找到已处理文件记录，将创建新记录")
    
    def _save_processed_files(self):
        """保存已处理文件记录"""
        try:
            with open(self.processed_files_file, 'w', encoding='utf-8') as f:
                for file_path in self.processed_files:
                    f.write(file_path + '\n')
        except Exception as e:
            print(f"保存已处理文件记录失败: {e}")
    
    def _get_file_hash(self, file_path):
        """计算文件哈希值，用于判断文件是否被修改"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None
    
    def _is_file_processed(self, file_path):
        """检查文件是否已被处理"""
        file_path_str = str(file_path)
        return file_path_str in self.processed_files
    
    def _mark_file_processed(self, file_path):
        """标记文件为已处理"""
        file_path_str = str(file_path)
        self.processed_files.add(file_path_str)
        self._save_processed_files()
    
    def _process_image(self, image_path):
        """处理单张图片"""
        if not self.model:
            print(f"模型未加载，无法处理图片: {image_path}")
            return False
        
        try:
            # 读取图片
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"无法读取图片: {image_path}")
                return False
            
            # 进行预测
            results = self.model(image, conf=0.6)
            
            # 初始化结果记录
            has_water = False
            water_count = 0
            confidence = 0.0
            water_area_ratio = 0.0  # 积水区域占比
            
            # 创建可视化图片
            vis_image = image.copy()
            
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    boxes = result.boxes.data.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    
                    water_count = len(masks)
                    if water_count > 0:
                        has_water = True
                        confidence = float(np.mean(confs))
                    
                    # 计算积水区域总面积
                    total_water_area = 0
                    image_area = image.shape[0] * image.shape[1]  # 图片总面积
                    
                    # 绘制每个检测到的区域
                    for mask, box, conf in zip(masks, boxes, confs):
                        # 检查并调整掩码尺寸
                        if mask.shape[:2] != image.shape[:2]:
                            mask = cv2.resize(mask.astype(np.float32), (image.shape[1], image.shape[0]))
                            mask = (mask > 0.5).astype(np.uint8)
                        
                        # 计算当前掩码的面积
                        mask_area = np.sum(mask > 0)
                        total_water_area += mask_area
                        
                        # 找到轮廓
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # 绘制轮廓（绿色，线宽2）
                        cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)
                        
                        # 绘制边界框
                        x1, y1, x2, y2 = map(int, box[:4])
                        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 计算积水区域占比
                    if image_area > 0 and has_water:
                        water_area_ratio = (total_water_area / image_area) * 100 * 1.3
                        
                        # 如果积水面积占比低于5%，认为没有积水
                        if water_area_ratio < 5:
                            has_water = False
                            water_count = 0
            
            # 创建结果文件夹
            result_folder = image_path.parent / "积水识别结果"
            result_folder.mkdir(exist_ok=True)
            
            # 保存识别后的图片
            base_name = image_path.stem
            save_path = result_folder / f"{base_name}_积水识别.png"
            cv2.imwrite(str(save_path), vis_image)
            
            # 记录结果到CSV
            self._save_result_to_csv(image_path, save_path, has_water, water_count, confidence, water_area_ratio)
            
            print(f"✓ 已处理: {image_path.name} -> {save_path.name}")
            return True
            
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {e}")
            return False
    
    def _save_result_to_csv(self, image_path, save_path, has_water, water_count, confidence, water_area_ratio):
        """保存识别结果到CSV文件"""
        try:
            # 为每个子文件夹创建单独的CSV文件
            subfolder = image_path.parent
            csv_path = subfolder / f"{subfolder.name}_积水识别结果.csv"
            
            # 检查CSV文件是否存在，如果不存在则创建表头
            file_exists = csv_path.exists()
            
            with open(csv_path, 'a', newline='', encoding='utf-8-sig') as csvfile:
                fieldnames = ['原图片名称', '原图片路径', '识别后图片名称', 
                             '识别后图片路径', '是否有积水', '积水区域数量', '平均置信度', '积水面积占比(%)', '处理时间']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                result_record = {
                    '原图片名称': image_path.name,
                    '原图片路径': str(image_path),
                    '识别后图片名称': save_path.name,
                    '识别后图片路径': str(save_path),
                    '是否有积水': '是' if has_water else '否',
                    '积水区域数量': water_count,
                    '平均置信度': round(confidence, 3),
                    '积水面积占比(%)': round(water_area_ratio, 2),
                    '处理时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                writer.writerow(result_record)
                
        except Exception as e:
            print(f"保存CSV结果失败: {e}")
    
    def _scan_and_process_new_files(self):
        """扫描并处理新文件"""
        with self.processing_lock:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始扫描新文件...")
            
            # 获取所有子文件夹
            subfolders = [f for f in self.root_folder.iterdir() 
                         if f.is_dir() and not f.name.startswith('.') and not f.name == "积水识别结果"]
            
            new_files_count = 0
            
            for subfolder in subfolders:
                # 获取子文件夹中的所有图片
                image_files = [f for f in subfolder.iterdir() 
                              if f.is_file() and f.suffix.lower() in self.image_extensions]
                
                for image_file in image_files:
                    if not self._is_file_processed(image_file):
                        print(f"发现新文件: {image_file}")
                        if self._process_image(image_file):
                            self._mark_file_processed(image_file)
                            new_files_count += 1
            
            if new_files_count > 0:
                print(f"本次扫描处理了 {new_files_count} 个新文件")
            else:
                print("本次扫描未发现新文件")
    
    def on_created(self, event):
        """文件创建事件"""
        if not event.is_directory and event.src_path.lower().endswith(tuple(self.image_extensions)):
            print(f"检测到新图片文件: {event.src_path}")
            # 延迟处理，确保文件写入完成
            time.sleep(1)
            self._scan_and_process_new_files()
    
    def on_moved(self, event):
        """文件移动事件"""
        if not event.is_directory and event.dest_path.lower().endswith(tuple(self.image_extensions)):
            print(f"检测到图片文件移动: {event.dest_path}")
            time.sleep(1)
            self._scan_and_process_new_files()
    
    def start_monitoring(self):
        """开始监控"""
        # 初始扫描
        self._scan_and_process_new_files()
        
        # 设置定时扫描
        def periodic_scan():
            while True:
                time.sleep(self.check_interval)
                self._scan_and_process_new_files()
        
        # 启动定时扫描线程
        scan_thread = threading.Thread(target=periodic_scan, daemon=True)
        scan_thread.start()
        
        print("监控已启动，按 Ctrl+C 停止...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n正在停止监控...")
            self._save_processed_files()
            print("监控已停止")


def process_all_subfolders(model_path, root_folder):
    """
    处理根文件夹下所有子文件夹的积水识别（一次性处理所有文件）
    
    Args:
        model_path: 模型路径
        root_folder: 根文件夹路径
    """
    # 加载模型
    print("正在加载模型...")
    model = YOLO(model_path)
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 获取所有子文件夹
    root_path = Path(root_folder)
    subfolders = [f for f in root_path.iterdir() if f.is_dir() and not f.name.startswith('.')]
    
    print(f"找到 {len(subfolders)} 个子文件夹")
    
    # 记录所有结果的列表
    all_results = []
    
    # 处理每个子文件夹
    for subfolder in tqdm(subfolders, desc="处理子文件夹"):
        print(f"\n正在处理子文件夹: {subfolder.name}")
        
        # 在子文件夹中创建结果文件夹
        result_folder = subfolder / "积水识别结果"
        result_folder.mkdir(exist_ok=True)
        
        # 获取子文件夹中的所有图片
        image_files = [f for f in subfolder.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        print(f"  找到 {len(image_files)} 张图片")
        
        # 处理每张图片
        for image_file in tqdm(image_files, desc=f"处理 {subfolder.name} 中的图片"):
            try:
                # 读取图片
                image = cv2.imread(str(image_file))
                if image is None:
                    print(f"  无法读取图片: {image_file}")
                    continue
                
                # 进行预测
                results = model(image, conf=0.6, verbose=False)
                
                # 初始化结果记录
                has_water = False
                water_count = 0
                confidence = 0.0
                water_area_ratio = 0.0  # 积水区域占比
                
                # 初始化可视化图片
                vis_image = image.copy()
                            
                if results and len(results) > 0:
                    result = results[0]
                    
                    if hasattr(result, 'masks') and result.masks is not None:
                        masks = result.masks.data.cpu().numpy()
                        boxes = result.boxes.data.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()
                        
                        water_count = len(masks)
                        if water_count > 0:
                            has_water = True
                            confidence = float(np.mean(confs))
                        
                        # 计算积水区域总面积
                        total_water_area = 0
                        image_area = image.shape[0] * image.shape[1]  # 图片总面积
                        
                        # 绘制每个检测到的区域
                        for mask, box, conf in zip(masks, boxes, confs):
                            # 检查并调整掩码尺寸
                            if mask.shape[:2] != image.shape[:2]:
                                mask = cv2.resize(mask.astype(np.float32), (image.shape[1], image.shape[0]))
                                mask = (mask > 0.5).astype(np.uint8)
                            
                            # 计算当前掩码的面积
                            mask_area = np.sum(mask > 0)
                            total_water_area += mask_area
                            
                            # 找到轮廓
                            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            # 绘制轮廓（绿色，线宽2）
                            cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)
                            
                            # 绘制边界框
                            x1, y1, x2, y2 = map(int, box[:4])
                            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 计算积水区域占比
                        if image_area > 0 and has_water:
                            water_area_ratio = (total_water_area / image_area) * 100 * 1.3
                            
                            # 如果积水面积占比低于5%，认为没有积水
                            if water_area_ratio < 5:
                                has_water = False
                                water_count = 0
                
                # 保存识别后的图片
                base_name = image_file.stem
                save_path = result_folder / f"{base_name}_积水识别.png"
                cv2.imwrite(str(save_path), vis_image)
                
                # 记录结果
                result_record = {
                    '子文件夹': subfolder.name,
                    '原图片名称': image_file.name,
                    '原图片路径': str(image_file),
                    '识别后图片名称': f"{base_name}_积水识别.png",
                    '识别后图片路径': str(save_path),
                    '是否有积水': '是' if has_water else '否',
                    '积水区域数量': water_count,
                    '平均置信度': round(confidence, 3),
                    '积水面积占比(%)': round(water_area_ratio, 2),
                    '处理时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                all_results.append(result_record)
                
            except Exception as e:
                print(f"  处理图片 {image_file} 时出错: {e}")
                # 记录错误信息
                error_record = {
                    '子文件夹': subfolder.name,
                    '原图片名称': image_file.name,
                    '原图片路径': str(image_file),
                    '识别后图片名称': '处理失败',
                    '识别后图片路径': '处理失败',
                    '是否有积水': '处理失败',
                    '积水区域数量': 0,
                    '平均置信度': 0.0,
                    '积水面积占比(%)': 0.0,
                    '处理时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                all_results.append(error_record)
                continue
        
        # 为每个子文件夹创建单独的CSV文件
        csv_path = subfolder / f"{subfolder.name}_积水识别结果.csv"
        create_csv_for_subfolder(all_results, csv_path, subfolder.name)
    
    # 创建总的CSV文件
    total_csv_path = root_path / "所有子文件夹积水识别结果汇总.csv"
    create_total_csv(all_results, total_csv_path)
    
    print(f"\n处理完成！")
    print(f"总共处理了 {len(all_results)} 张图片")
    print(f"汇总结果保存在: {total_csv_path}")


def create_csv_for_subfolder(results, csv_path, subfolder_name):
    """
    为单个子文件夹创建CSV文件
    """
    # 筛选当前子文件夹的结果
    subfolder_results = [r for r in results if r['子文件夹'] == subfolder_name]
    
    if not subfolder_results:
        return
    
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['子文件夹', '原图片名称', '原图片路径', '识别后图片名称', 
                     '识别后图片路径', '是否有积水', '积水区域数量', '平均置信度', '积水面积占比(%)', '处理时间']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in subfolder_results:
            writer.writerow(result)
    
    print(f"  CSV文件已保存: {csv_path}")


def create_total_csv(results, csv_path):
    """
    创建总的CSV文件
    """
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['子文件夹', '原图片名称', '原图片路径', '识别后图片名称', 
                     '识别后图片路径', '是否有积水', '积水区域数量', '平均置信度', '积水面积占比(%)', '处理时间']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"总CSV文件已保存: {csv_path}")


def auto_detect_new_folders(root_folder, processed_folders_file):
    """
    自动检测新文件夹
    """
    processed_folders_file = Path(processed_folders_file)
    
    # 读取已处理的文件夹列表
    processed_folders = set()
    if processed_folders_file.exists():
        with open(processed_folders_file, 'r', encoding='utf-8') as f:
            processed_folders = set(line.strip() for line in f if line.strip())
    
    # 获取当前所有子文件夹
    root_path = Path(root_folder)
    current_folders = {f.name for f in root_path.iterdir() 
                      if f.is_dir() and not f.name.startswith('.')}
    
    # 找出新文件夹
    new_folders = current_folders - processed_folders
    
    if new_folders:
        print(f"发现新文件夹: {', '.join(new_folders)}")
        return True
    else:
        print("没有发现新文件夹")
        return False


if __name__ == "__main__":
    # 配置参数（直接修改这里）
    MODEL_PATH = WATER_SEGMENT_WEIGHTS  # 你的模型文件路径
    ROOT_FOLDER = RUNS_OUTPUT_IMAGES_DIR  # 根文件夹路径
    PROCESSED_FILES_FILE = PROCESSED_FILES_PATH  # 记录已处理文件的文件
    CHECK_INTERVAL = 30  # 定时扫描间隔（秒）
    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"错误：模型文件不存在: {MODEL_PATH}")
        print("请修改 MODEL_PATH 为正确的模型文件路径")
        exit()
    
    # 检查根文件夹
    if not os.path.exists(ROOT_FOLDER):
        print(f"错误：根文件夹不存在: {ROOT_FOLDER}")
        print("请修改 ROOT_FOLDER 为正确的根文件夹路径")
        exit()
    
    print("积水识别监控系统")
    print("=" * 50)
    print("1. 启动文件夹监控（自动检测新文件）")
    print("2. 一次性处理所有现有文件")
    print("3. 退出")
    
    while True:
        try:
            choice = input("\n请选择操作 (1/2/3): ").strip()
            
            if choice == "1":
                print("\n启动文件夹监控模式...")
                monitor = WaterDetectionMonitor(MODEL_PATH, ROOT_FOLDER, PROCESSED_FILES_FILE, CHECK_INTERVAL)
                monitor.start_monitoring()
                break
                
            elif choice == "2":
                print("\n启动一次性处理模式...")
                process_all_subfolders(MODEL_PATH, ROOT_FOLDER)
                break
                
            elif choice == "3":
                print("退出程序")
                break
                
            else:
                print("无效选择，请输入 1、2 或 3")
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            break
