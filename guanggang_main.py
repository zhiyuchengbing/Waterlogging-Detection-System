import importlib
import types
from collections import defaultdict
import os
import time
import datetime
import shutil
from pathlib import Path
import hashlib
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from tqdm import tqdm
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton,
                               QVBoxLayout, QHBoxLayout, QLabel, QWidget, QFileDialog,
                               QDialog, QGridLayout, QDoubleSpinBox, QListWidget,
                               QScrollArea, QSplitter, QMessageBox, QListWidgetItem,
                               QCheckBox, QDateTimeEdit, QPushButton,
                               QMenu, QGroupBox, QComboBox, QLineEdit, QTableWidget,
                               QTableWidgetItem, QHeaderView, QFrame, QDateEdit,
                               QTextEdit, QProgressBar, QStatusBar, QSpacerItem,
                               QSizePolicy, QAbstractItemView)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QDateTime, QFileInfo, QTimer, QDate
from PySide6.QtGui import QImage, QPixmap, QColor, QBrush, QIcon, QAction, QFont
import sys
import subprocess
import platform
import os.path
import re
import glob
import csv
import warnings
warnings.filterwarnings('ignore')

# 固定输出目录为绝对路径，避免工程目录移动导致历史数据路径变化
OUTPUT_IMAGES_DIR = r"E:\积水识别项目\demo0625\demo\output_images"
RUNS_OUTPUT_IMAGES_DIR = r"E:\积水识别项目\demo0625\demo\runs\output_images"

# 固定数据与状态目录为绝对路径
DATA_DIR = r"E:\积水识别项目\demo0625\demo\data"
WATER_STATUS_DIR = r"E:\积水识别项目\demo0625\demo\water_detection_status"
PROCESSED_FILES_PATH = r"E:\积水识别项目\demo0625\demo\已处理文件记录.txt"

# 固定模型权重绝对路径，避免工程搬迁后相对路径失效
SEGMENT_TRAIN2_WEIGHTS = r"E:\积水识别项目\demo0625\demo\runs\segment\train2\weights\best.pt"
WATER_SEGMENT_WEIGHTS = r"E:\积水识别项目\demo0625\demo\runs\segment\train\weights_jishui\best4.pt"
DETECT_TRAIN5_WEIGHTS = r"E:\积水识别项目\demo0625\demo\runs\detect\train5\weights\best.pt"
DETECT_TRAIN2_PLATE_WEIGHTS = r"E:\积水识别项目\demo0625\demo\runs\detect\train2\weights\best3_0816.pt"


class LazyLoader(types.ModuleType):
    """Lazy import wrapper to defer heavy module loading until first use."""

    def __init__(self, local_name, parent_globals, name):
        super().__init__(name)
        self._local_name = local_name
        self._parent_globals = parent_globals
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = importlib.import_module(self.__name__)
            self._parent_globals[self._local_name] = self._module
        return self._module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)


cv2 = LazyLoader("cv2", globals(), "cv2")
np = LazyLoader("np", globals(), "numpy")
pd = LazyLoader("pd", globals(), "pandas")
plt = LazyLoader("plt", globals(), "matplotlib.pyplot")
mdates = LazyLoader("mdates", globals(), "matplotlib.dates")
sns = LazyLoader("sns", globals(), "seaborn")
Image = LazyLoader("Image", globals(), "PIL.Image")

_YOLO_CLASS = None


def load_yolo_class():
    """Lazily import the YOLO class from ultralytics when needed."""

    global _YOLO_CLASS
    if _YOLO_CLASS is None:
        from ultralytics import YOLO as _YOLO

        _YOLO_CLASS = _YOLO
    return _YOLO_CLASS


_matplotlib_configured = False


def ensure_matplotlib_configured():
    """Apply matplotlib font configuration once before plotting."""

    global _matplotlib_configured
    if _matplotlib_configured:
        return
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 10

        try:
            import matplotlib.font_manager as fm

            font_list = [f.name for f in fm.fontManager.ttflist]
            chinese_fonts = [f for f in font_list if any(ch in f for ch in ['Microsoft YaHei', 'SimHei', '微软雅黑', '黑体'])]
            if chinese_fonts:
                plt.rcParams['font.family'] = chinese_fonts[0]
                print(f"使用中文字体: {chinese_fonts[0]}")
        except Exception:
            print("使用默认字体设置")
    except Exception as exc:
        print(f"初始化matplotlib失败: {exc}")
        return

    _matplotlib_configured = True

# 导入准确率可视化模块
try:
    from accuracy_visualization import TrainAccuracyMainWindow
except ImportError:
    TrainAccuracyMainWindow = None
    print("警告：未找到 accuracy_visualization.py 模块，准确率可视化功能将不可用")

# 积水识别相关模块改为延迟导入，以加快启动速度
# from inference2 import WaterDetectionMonitor, process_all_subfolders

# 导入数据库自动更新器
from database_auto_updater import DatabaseAutoUpdater

# 确保当前目录加入sys.path，便于本地模块导入
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
except Exception:
    pass

# 导入MySQL查询/审核函数（打印真实错误便于排查依赖）
try:
    from test_mysql import (
        query_recent_water_carriages,
        review_mark_no_water,
        review_confirm_water,
    )
except Exception as e:
    query_recent_water_carriages = None
    review_mark_no_water = None
    review_confirm_water = None
    print(f"警告：导入 test_mysql 失败：{e}")


class AutoProcessor(QThread):
    status_changed = Signal(str, str)  # 发送视频路径和状态的信号
    processing_completed = Signal(str)  # 发送处理完成的视频路径信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.video_folder = r"E:\积水识别项目\视频下载模块\record"  # 设置默认值为指定路径
        self.processed_videos = set()  # 保留此集合，但不再使用它来判断是否处理过
        self.processing_video = None
        self.pending_videos = set()
        self.last_frame_time = 0
        self.scan_interval = 30  # 扫描间隔（秒），适当调大以减少频繁全盘扫描
        self.timeout = 10  # 无新帧超时时间（秒）

    def set_video_folder(self, folder):
        self.video_folder = folder
        self.scan_videos()

    def scan_videos(self):
        """扫描文件夹中的视频文件"""
        if not self.video_folder:
            return

        # 获取所有36开头的视频文件
        video_pattern = os.path.join(self.video_folder, "**", "36*.mp4")
        video_files = glob.glob(video_pattern, recursive=True)

        # 只保留最近一周内的新视频，避免对很久以前的历史数据反复扫描
        one_week_ago = time.time() - 7 * 24 * 60 * 60
        recent_video_files = []
        for video in video_files:
            try:
                mtime = os.path.getmtime(video)
                if mtime >= one_week_ago:
                    recent_video_files.append(video)
            except OSError:
                continue

        # 更新待处理列表
        new_pending_videos = set()
        for video in recent_video_files:
            if not self.is_video_processed(video):
                new_pending_videos.add(video)

        # 添加新发现的待处理视频
        for video in new_pending_videos - self.pending_videos:
            self.status_changed.emit(video, "待处理")

        # 更新待处理列表，保留新发现的和已有的但未处理的视频
        self.pending_videos = new_pending_videos
        if self.processing_video:
            self.pending_videos.discard(self.processing_video)

    def is_video_processed(self, video_path):
        """检查视频是否已处理"""
        parent_folder = os.path.basename(os.path.dirname(video_path))
        result_dir = os.path.join(RUNS_OUTPUT_IMAGES_DIR, parent_folder)

        # 检查结果目录是否存在
        if not os.path.exists(result_dir):
            return False

        # 检查是否存在处理完成标记文件
        processed_marker = os.path.join(result_dir, "processed_complete.txt")
        if os.path.exists(processed_marker):
            return True
            
        # 检查结果目录中是否有图片文件（兼容旧版本）
        image_files = glob.glob(os.path.join(result_dir, "frame_*.jpg"))
        return len(image_files) > 0  # 如果有图片文件，则认为已处理过

    def run(self):
        self.running = True
        while self.running:
            # 仅在没有正在处理的视频时才扫描新视频，避免频繁全盘扫描
            if not self.processing_video:
                self.scan_videos()

                # 如果有待处理的视频，取下一个开始处理
                if self.pending_videos:
                    self.processing_video = next(iter(self.pending_videos))
                    self.status_changed.emit(self.processing_video, "正在处理")

            time.sleep(self.scan_interval)

    def stop(self):
        self.running = False
        self.wait()

    def update_last_frame_time(self):
        """更新最后处理帧的时间"""
        self.last_frame_time = time.time()

    def video_finished(self, video_path):
        """视频处理完成回调"""
        if self.processing_video == video_path:
            # 标记为已处理并移动到下一个
            self.processed_videos.add(video_path)
            self.processing_completed.emit(video_path)
            self.processing_video = None

            # 重新扫描，更新状态
            self.scan_videos()


class VideoThread(QThread):
    change_pixmap_signal = Signal(np.ndarray)
    count_signal = Signal(int)
    saved_image_signal = Signal(str)  # 用于发送保存的图片路径信号
    fps_signal = Signal(float)  # 用于发送帧率信号
    video_finished_signal = Signal(str)  # 添加视频处理完成信号

    def __init__(self):
        super().__init__()
        
        self.running = False
        self.model = None
        self.save_path = RUNS_OUTPUT_IMAGES_DIR  # 默认保存路径

        self.conf_threshold = 0.6  # 默认置信度阈值
        # 监测区域设置，默认为全屏
        self.detection_area_enabled = False
        self.left_boundary_ratio = 1 / 3  # 左边界位于屏幕宽度的1/3处
        self.right_boundary_ratio = 2 / 3  # 右边界位于屏幕宽度的2/3处
        # 修改计数范围设置
        self.count_line_top = 700  # 计数区域上边界
        self.count_line_bottom = 800  # 计数区域下边界 --------------------------------------------------------------
        # 添加已计数ID集合
        self.counted_ids = set()
        # 添加目标状态追踪字典
        self.target_states = {}  # 记录每个ID的上一帧位置状态
        # 添加目标运动方向追踪
        self.target_directions = {}  # 记录每个ID的运动方向：'up' 或 'down'
        # 添加目标进入计数区域的状态追踪
        self.target_entered_counting_area = {}  # 记录每个ID是否已进入计数区域
        # 添加内存管理参数
        self.gc_interval = 20  # 每处理20帧执行一次垃圾回收
        self.frame_count = 0
        # 添加超时控制参数
        self.max_processing_time = 3 * 60 * 60  # 3小时的秒数
        self.start_time = None  # 处理开始时间

    def reset_state(self):
        """重置所有状态"""
        self.counted_ids.clear()  # 清空已计数ID集合
        self.target_states.clear()  # 清空目标状态字典
        self.target_directions.clear()  # 清空目标运动方向字典
        self.target_entered_counting_area.clear()  # 清空目标进入计数区域状态字典
        self.count_signal.emit(0)  # 重置计数显示
        self.frame_count = 0  # 重置帧计数
        self.start_time = None  # 重置开始时间

    def set_model_path(self, model_path):
        yolo_cls = load_yolo_class()
        self.model = yolo_cls(model_path)

    def set_save_path(self, save_path):
        self.save_path = save_path

    def set_video_path(self, video_path):
        self.video_path = video_path
        # 设置新视频时重置状态
        self.reset_state()

    def set_conf_threshold(self, conf):
        self.conf_threshold = conf

    def set_detection_area(self, enabled):
        self.detection_area_enabled = enabled

    def create_box_only_frame(self, frame, box):
        """创建只包含边界框的图片，不显示ID、掩膜和辅助线"""
        # 复制原始帧
        box_only_frame = frame.copy()

        # 解析边界框坐标
        x, y, w, h = box

        # 绘制边界框，绿色，线宽2
        cv2.rectangle(box_only_frame,
                      (int(x - w / 2), int(y - h / 2)),
                      (int(x + w / 2), int(y + h / 2)),
                      (0, 255, 0), 2)

        return box_only_frame

    def run(self):
        self.running = True
        count = 0
        
        # 设置处理开始时间
        self.start_time = time.time()
        
        try:
            capture = cv2.VideoCapture(self.video_path)

            # 检查视频是否成功打开
            if not capture.isOpened():
                print(f"错误：无法打开视频文件 {self.video_path}")
                self.video_finished_signal.emit(self.video_path)
                return

            # 使用视频文件所在的父文件夹名称创建保存目录
            parent_folder = os.path.basename(os.path.dirname(self.video_path))
            self.current_save_path = os.path.join(self.save_path, parent_folder)
            os.makedirs(self.current_save_path, exist_ok=True)

            # 帧率计算变量
            frame_times = []
            fps_update_interval = 10  # 每10帧更新一次FPS

            while self.running:
                # 检查是否超时（3小时）
                current_time = time.time()
                if current_time - self.start_time > self.max_processing_time:
                    print(f"视频处理超时（超过3小时），跳过处理: {self.video_path}")
                    break
                
                start_time = time.time()  # 开始计时

                try:
                    ret, frame = capture.read()
                    if not ret:
                        break

                    # 更新帧计数并定期执行垃圾回收
                    self.frame_count += 1
                    if self.frame_count % self.gc_interval == 0:
                        # 执行垃圾回收
                        import gc
                        gc.collect()

                    frame_height = frame.shape[0]
                    frame_width = frame.shape[1]

                    # 计算监测区域的边界
                    left_boundary = int(frame_width * self.left_boundary_ratio)
                    right_boundary = int(frame_width * self.right_boundary_ratio)

                    # 如果启用了区域监测，只处理中间区域
                    if self.detection_area_enabled:
                        # 创建感兴趣区域掩码
                        mask = np.zeros_like(frame)
                        # 只在竖线之间的区域设置为有效
                        mask[:, left_boundary:right_boundary, :] = 255
                        # 应用掩码，其他区域置黑
                        frame_masked = cv2.bitwise_and(frame, mask)
                        # 处理掩码区域
                        frame_rgb = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2RGB)
                        del frame_masked  # 释放内存
                        del mask  # 释放内存
                    else:
                        # 如果不启用区域监测，处理整个帧
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # 使用模型处理帧
                    results = self.model.track(frame_rgb, conf=self.conf_threshold, iou=0.1, persist=True, batch=64)
                    del frame_rgb  # 释放不再需要的内存

                    # 复制原始帧用于绘制，保留完整画面
                    a_frame = frame.copy()
                    if len(results) > 0 and results[0].boxes is not None:
                        # 在复制的帧上绘制检测结果
                        a_frame = results[0].plot(img=a_frame)
                        ids = results[0].boxes.id

                        # 在帧上绘制计数区域
                        cv2.line(a_frame, (0, self.count_line_top), (frame_width, self.count_line_top), (0, 255, 0), 2)
                        cv2.line(a_frame, (0, self.count_line_bottom), (frame_width, self.count_line_bottom),
                                 (0, 255, 0), 2)

                        # 在帧上绘制竖向边界线(如果启用了区域监测)
                        if self.detection_area_enabled:
                            cv2.line(a_frame, (left_boundary, 0), (left_boundary, frame_height), (255, 0, 0), 2)
                            cv2.line(a_frame, (right_boundary, 0), (right_boundary, frame_height), (255, 0, 0), 2)

                        if ids is not None:
                            ids = ids.int().cpu().tolist()
                            boxes = results[0].boxes.xywh.cpu()

                            for tid, box in zip(ids, boxes):
                                x, y, w, h = box

                                # 只检查位于中间区域的对象
                                if self.detection_area_enabled:
                                    in_detection_area = left_boundary <= x <= right_boundary
                                else:
                                    in_detection_area = True

                                # 检查是否在计数区域内且未计数过
                                if in_detection_area and tid not in self.counted_ids:
                                    # 检查是否在计数区域内
                                    if self.count_line_top <= y <= self.count_line_bottom:
                                        # 简化计数逻辑：只要ID进入监测区域就计数，不考虑方向
                                        # 列车是单向的，同一ID不会再次出现
                                        count += 1
                                        self.counted_ids.add(tid)
                                        self.count_signal.emit(count)

                                        # 记录方向（仅用于显示统计）
                                        if tid in self.target_states:
                                            prev_y = self.target_states[tid]
                                            if y > prev_y:
                                                self.target_directions[tid] = 'down'
                                            elif y < prev_y:
                                                self.target_directions[tid] = 'up'
                                        else:
                                            # 新目标，根据位置在计数区域中的位置判断方向（仅用于显示）
                                            if y < (self.count_line_top + self.count_line_bottom) / 2:
                                                self.target_directions[tid] = 'down'
                                            else:
                                                self.target_directions[tid] = 'up'

                                        # 创建只包含边界框的图片用于保存
                                        try:
                                            # 提取目标区域而不是整个帧
                                            x1, y1 = max(0, int(x - w / 2)), max(0, int(y - h / 2))
                                            x2, y2 = min(frame_width, int(x + w / 2)), min(frame_height, int(y + h / 2))
                                            target_crop = frame[y1:y2, x1:x2].copy()

                                            # 保存裁剪后的目标图像
                                            frame_save_path = os.path.join(self.current_save_path, f"frame_{count}.jpg")
                                            cv2.imwrite(frame_save_path, target_crop)

                                            # 发送保存图片路径信号
                                            self.saved_image_signal.emit(frame_save_path)
                                            del target_crop  # 释放内存
                                        except Exception as e:
                                            print(f"保存图像时出错: {str(e)}")

                                # 更新目标状态
                                self.target_states[tid] = y
                    else:
                        # 如果没有检测到对象，仍然绘制辅助线
                        cv2.line(a_frame, (0, self.count_line_top), (frame_width, self.count_line_top), (0, 255, 0), 2)
                        cv2.line(a_frame, (0, self.count_line_bottom), (frame_width, self.count_line_bottom),
                                 (0, 255, 0), 2)

                        if self.detection_area_enabled:
                            cv2.line(a_frame, (left_boundary, 0), (left_boundary, frame_height), (255, 0, 0), 2)
                            cv2.line(a_frame, (right_boundary, 0), (right_boundary, frame_height), (255, 0, 0), 2)

                    # 显示计数
                    cv2.putText(a_frame, f"计数: {count}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # 显示运动方向信息
                    up_count = sum(1 for direction in self.target_directions.values() if direction == 'up')
                    down_count = sum(1 for direction in self.target_directions.values() if direction == 'down')
                    cv2.putText(a_frame, f"向上: {up_count} 向下: {down_count}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    # 对UI显示的图像进行优化，保持高度比例适应宽度800像素
                    h, w = a_frame.shape[:2]
                    display_w = min(800, w)  # 限制最大宽度为800
                    display_h = int(h * (display_w / w))
                    if h > 600 or w > 800:
                        display_frame = cv2.resize(a_frame, (display_w, display_h), interpolation=cv2.INTER_AREA)
                        self.change_pixmap_signal.emit(display_frame)
                        del display_frame  # 释放内存
                    else:
                        # 如果尺寸已经够小，直接使用
                        self.change_pixmap_signal.emit(a_frame)

                    del a_frame  # 释放内存
                    del frame  # 释放原始帧内存

                    # 计算处理一帧所需的时间
                    end_time = time.time()
                    frame_time = end_time - start_time
                    frame_times.append(frame_time)

                    # 每隔一定帧数更新一次FPS
                    if len(frame_times) >= fps_update_interval:
                        avg_frame_time = sum(frame_times) / len(frame_times)
                        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                        self.fps_signal.emit(fps)
                        # 只保留最近的帧时间用于计算
                        frame_times = frame_times[-fps_update_interval:]

                except cv2.error as e:
                    print(f"OpenCV错误: {str(e)}")
                    continue
                except MemoryError:
                    print("内存不足，尝试释放资源")
                    # 强制执行垃圾回收
                    import gc
                    gc.collect()
                    continue
                except Exception as e:
                    print(f"处理帧时出错: {str(e)}")
                    continue

        except Exception as e:
            print(f"视频处理线程出现错误: {str(e)}")
        finally:
            # 确保在任何情况下都释放资源
            if 'capture' in locals() and capture is not None:
                capture.release()
                
            # 创建处理完成标记文件，避免重复处理
            if hasattr(self, 'current_save_path') and self.current_save_path:
                try:
                    os.makedirs(self.current_save_path, exist_ok=True)
                    processed_marker = os.path.join(self.current_save_path, "processed_complete.txt")
                    with open(processed_marker, 'w', encoding='utf-8') as f:
                        f.write(f"视频处理完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"视频路径: {self.video_path}\n")
                        f.write(f"检测到的对象数量: {count}\n")
                    print(f"已创建处理完成标记文件: {processed_marker}")
                except Exception as e:
                    print(f"创建处理完成标记文件失败: {str(e)}")

            # 发送视频处理完成信号
            self.video_finished_signal.emit(self.video_path)

    def stop(self):
        self.running = False
        self.wait()


# Cemian系统视频处理线程
class CemianVideoThread(QThread):
    change_pixmap_signal = Signal(np.ndarray)
    count_signal = Signal(int)
    count_right_to_left_signal = Signal(int)  # 从右往左计数信号
    count_left_to_right_signal = Signal(int)  # 从左往右计数信号
    count_line1_signal = Signal(int)  # 第一条线计数信号
    count_line2_signal = Signal(int)  # 第二条线计数信号
    count_line3_signal = Signal(int)  # 第三条线计数信号
    count_line4_signal = Signal(int)  # 第四条线计数信号
    count_line5_signal = Signal(int)  # 第五条线计数信号
    saved_image_signal = Signal(str)  # 用于发送保存的图片路径信号
    fps_signal = Signal(float)

    def __init__(self):
        super().__init__()
        self.running = False
        self.auto_processing = False
        self.model = None
        self.plate_model = None
        self.save_path = OUTPUT_IMAGES_DIR
        self.video_folder = r"E:\积水识别项目\视频下载模块\record"
        self.conf_threshold = 0.3
        self.processed_videos = set()
        self.current_video = None
        self.crop_box = (500, 300, 1800, 750)  # 添加裁切区域
        self.crop_box_second = (900, 300, 2200, 750)  # 第二条线的裁切区域（向右平移400像素）
        self.crop_box_third = (1300, 300, 2600, 750)  # 第三条线的裁切区域（继续向右平移400像素）
        self.crop_box_fourth = (700, 300, 2000, 750)  # 第四条线的裁切区域（左-中间之间）
        self.crop_box_fifth = (1100, 300, 2400, 750)  # 第五条线的裁切区域（中-右之间）

    def set_model_path(self, model_path):
        try:
            model_path = os.path.normpath(model_path)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            yolo_cls = load_yolo_class()
            self.model = yolo_cls(model_path)
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            raise

    def set_plate_model_path(self, model_path):
        """设置车牌识别模型路径"""
        try:
            model_path = os.path.normpath(model_path)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"车牌识别模型文件不存在: {model_path}")
            yolo_cls = load_yolo_class()
            self.plate_model = yolo_cls(model_path)
        except Exception as e:
            print(f"加载车牌识别模型失败: {str(e)}")
            raise

    def set_save_path(self, save_path):
        self.save_path = save_path

    def set_conf_threshold(self, conf):
        self.conf_threshold = conf

    def start_auto_processing(self, video_folder):
        """开始自动处理"""
        self.video_folder = video_folder
        self.auto_processing = True
        self.running = True
        if not self.isRunning():
            self.start()

    def stop_auto_processing(self):
        """停止自动处理"""
        self.auto_processing = False
        self.running = False
        self.wait()

    def get_all_video_files(self, folder_path):
        """获取所有以33开头的视频文件"""
        video_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().startswith('33') and file.lower().endswith(('.mp4', '.avi')):
                    # 获取相对于根文件夹的路径
                    relative_path = os.path.relpath(os.path.join(root, file), folder_path)
                    video_files.append(relative_path)
        return video_files

    def run(self):
        """运行自动处理循环"""
        while self.running and self.auto_processing:
            try:
                # 获取所有视频文件
                video_files = self.get_all_video_files(self.video_folder)
                
                # 处理未处理的视频
                for video in video_files:
                    if not self.is_video_processed(video) and not self.current_video:
                        video_path = os.path.join(self.video_folder, video)
                        if os.path.exists(video_path):
                            self.current_video = video
                            self.process_video(video_path)
                            self.current_video = None
                            break
                
                # 如果没有需要处理的视频，短暂休眠
                if not self.current_video:
                    time.sleep(1)
                    
            except Exception as e:
                print(f"自动处理循环出错: {str(e)}")
                time.sleep(1)

    def is_video_processed(self, video_name):
        """检查视频是否已处理（与原始Cemian系统保持一致）"""
        # 构建视频完整路径以获取父文件夹名称
        video_path = os.path.join(self.video_folder, video_name)
        parent_folder = os.path.basename(os.path.dirname(video_path))
        output_video_dir = os.path.join(self.save_path, parent_folder)
        csv_path = os.path.join(output_video_dir, "plate_results.csv")

        # 检查目录是否存在且包含图片和CSV文件（与原始Cemian系统保持一致）
        is_processed = (os.path.exists(output_video_dir) and
                        os.path.isdir(output_video_dir) and
                        os.path.exists(csv_path) and
                        os.path.getsize(csv_path) > 0 and
                        any(f.endswith('.jpg') for f in os.listdir(output_video_dir)))
        
        return is_processed

    def process_video(self, video_path):
        """处理单个视频文件"""
        try:
            print(f"开始处理视频: {video_path}")
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = 0
            prev_x = None
            last_cross_time_line1 = 0
            last_cross_time_line2 = 0
            last_cross_time_line3 = 0
            last_cross_time_line4 = 0
            last_cross_time_line5 = 0
            count_right_to_left = 0
            count_left_to_right = 0
            frame_save_count = 0  # 添加帧保存计数器
            
            # 使用视频文件所在的父文件夹名称创建保存目录
            parent_folder = os.path.basename(os.path.dirname(video_path))
            video_save_dir = os.path.join(self.save_path, parent_folder)
            os.makedirs(video_save_dir, exist_ok=True)
            
            frame_times = []
            fps_update_interval = 10

            while self.running and self.auto_processing:
                start_time = time.time()
                success, frame = cap.read()
                if not success:
                    break

                frame_count += 1
                current_time = frame_count / fps if fps else 0

                # 获取帧的宽度以确定边界
                frame_width = frame.shape[1]
                center_line1 = 800  # 左侧检测线位置
                center_line2 = 1200  # 中间检测线位置
                # 以中间线为对称轴，在右侧新增一根检测线
                delta = center_line2 - center_line1
                center_line3 = center_line2 + delta  # 右侧检测线位置
                # 在左-中、 中-右 之间各增加一根辅助检测线
                center_line4 = (center_line1 + center_line2) // 2  # 左-中之间
                center_line5 = (center_line2 + center_line3) // 2  # 中-右之间
                edge_margin = 100  # 边缘区域宽度

                results = self.model(frame, conf=self.conf_threshold)
                boxes = results[0].boxes.xywh.cpu().numpy()
                annotated_frame = results[0].plot().copy()
                crossed_line1 = False
                crossed_line2 = False
                crossed_line3 = False
                crossed_line4 = False
                crossed_line5 = False

                if len(boxes) > 0:
                    x, y, w, h = boxes[0]
                    center_x = float(x)

                    # 检查物体是否在边缘区域
                    is_at_left_edge = center_x < edge_margin
                    is_at_right_edge = center_x > (frame_width - edge_margin)

                    # 如果是首次检测到物体
                    if prev_x is None:
                        prev_x = center_x
                    else:
                        # 检测穿越第一条线
                        time_diff_line1 = current_time - last_cross_time_line1
                        if time_diff_line1 > 0.5:  # 防止重复计数
                            # 只关注前一帧和当前帧位置与第一条中心线的关系
                            prev_at_right_line1 = prev_x > center_line1
                            curr_at_left_line1 = center_x < center_line1
                            prev_at_left_line1 = prev_x < center_line1
                            curr_at_right_line1 = center_x > center_line1

                            # 确保框真实穿越了第一条检测线（从右往左）
                            if prev_at_right_line1 and curr_at_left_line1:
                                count_right_to_left += 1
                                last_cross_time_line1 = current_time
                                crossed_line1 = True
                                self.count_right_to_left_signal.emit(count_right_to_left)
                                self.count_line1_signal.emit(count_right_to_left + count_left_to_right)

                            # 确保框真实穿越了第一条检测线（从左往右）
                            elif prev_at_left_line1 and curr_at_right_line1:
                                count_left_to_right += 1
                                last_cross_time_line1 = current_time
                                crossed_line1 = True
                                self.count_left_to_right_signal.emit(count_left_to_right)
                                self.count_line1_signal.emit(count_right_to_left + count_left_to_right)

                        # 检测穿越第二条线
                        time_diff_line2 = current_time - last_cross_time_line2
                        if time_diff_line2 > 0.5:  # 防止重复计数
                            # 只关注前一帧和当前帧位置与第二条中心线的关系
                            prev_at_right_line2 = prev_x > center_line2
                            curr_at_left_line2 = center_x < center_line2
                            prev_at_left_line2 = prev_x < center_line2
                            curr_at_right_line2 = center_x > center_line2

                            # 确保框真实穿越了第二条检测线（从右往左）
                            if prev_at_right_line2 and curr_at_left_line2:
                                count_right_to_left += 1
                                last_cross_time_line2 = current_time
                                crossed_line2 = True
                                self.count_right_to_left_signal.emit(count_right_to_left)
                                self.count_line2_signal.emit(count_right_to_left + count_left_to_right)

                            # 确保框真实穿越了第二条检测线（从左往右）
                            elif prev_at_left_line2 and curr_at_right_line2:
                                count_left_to_right += 1
                                last_cross_time_line2 = current_time
                                crossed_line2 = True
                                self.count_left_to_right_signal.emit(count_left_to_right)
                                self.count_line2_signal.emit(count_right_to_left + count_left_to_right)

                        # 检测穿越第三条线
                        time_diff_line3 = current_time - last_cross_time_line3
                        if time_diff_line3 > 0.5:  # 防止重复计数
                            # 只关注前一帧和当前帧位置与第三条中心线的关系
                            prev_at_right_line3 = prev_x > center_line3
                            curr_at_left_line3 = center_x < center_line3
                            prev_at_left_line3 = prev_x < center_line3
                            curr_at_right_line3 = center_x > center_line3

                            # 确保框真实穿越了第三条检测线（从右往左）
                            if prev_at_right_line3 and curr_at_left_line3:
                                count_right_to_left += 1
                                last_cross_time_line3 = current_time
                                crossed_line3 = True
                                self.count_right_to_left_signal.emit(count_right_to_left)
                                self.count_line3_signal.emit(count_right_to_left + count_left_to_right)

                            # 确保框真实穿越了第三条检测线（从左往右）
                            elif prev_at_left_line3 and curr_at_right_line3:
                                count_left_to_right += 1
                                last_cross_time_line3 = current_time
                                crossed_line3 = True
                                self.count_left_to_right_signal.emit(count_left_to_right)
                                self.count_line3_signal.emit(count_right_to_left + count_left_to_right)

                        # 检测穿越第四条线（左-中之间）
                        time_diff_line4 = current_time - last_cross_time_line4
                        if time_diff_line4 > 0.5:  # 防止重复计数
                            prev_at_right_line4 = prev_x > center_line4
                            curr_at_left_line4 = center_x < center_line4
                            prev_at_left_line4 = prev_x < center_line4
                            curr_at_right_line4 = center_x > center_line4

                            if prev_at_right_line4 and curr_at_left_line4:
                                count_right_to_left += 1
                                last_cross_time_line4 = current_time
                                crossed_line4 = True
                                self.count_right_to_left_signal.emit(count_right_to_left)
                                self.count_line4_signal.emit(count_right_to_left + count_left_to_right)
                            elif prev_at_left_line4 and curr_at_right_line4:
                                count_left_to_right += 1
                                last_cross_time_line4 = current_time
                                crossed_line4 = True
                                self.count_left_to_right_signal.emit(count_left_to_right)
                                self.count_line4_signal.emit(count_right_to_left + count_left_to_right)

                        # 检测穿越第五条线（中-右之间）
                        time_diff_line5 = current_time - last_cross_time_line5
                        if time_diff_line5 > 0.5:  # 防止重复计数
                            prev_at_right_line5 = prev_x > center_line5
                            curr_at_left_line5 = center_x < center_line5
                            prev_at_left_line5 = prev_x < center_line5
                            curr_at_right_line5 = center_x > center_line5

                            if prev_at_right_line5 and curr_at_left_line5:
                                count_right_to_left += 1
                                last_cross_time_line5 = current_time
                                crossed_line5 = True
                                self.count_right_to_left_signal.emit(count_right_to_left)
                                self.count_line5_signal.emit(count_right_to_left + count_left_to_right)
                            elif prev_at_left_line5 and curr_at_right_line5:
                                count_left_to_right += 1
                                last_cross_time_line5 = current_time
                                crossed_line5 = True
                                self.count_left_to_right_signal.emit(count_left_to_right)
                                self.count_line5_signal.emit(count_right_to_left + count_left_to_right)

                    prev_x = center_x

                # 处理穿越第一条线的照片保存
                if crossed_line1:
                    frame_save_count += 1
                    # 裁切图片（第一条线）
                    try:
                        # 将OpenCV图像转换为PIL图像
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        # 裁切图片（第一条线）
                        cropped = pil_image.crop(self.crop_box)
                        # 保存裁切后的图片
                        cropped_save_path = os.path.join(video_save_dir, f"{frame_save_count}_line1.jpg")
                        cropped.save(cropped_save_path)
                        # 发送裁切后的图片路径信号
                        self.saved_image_signal.emit(cropped_save_path)
                        # 对裁切后的图片进行车牌识别
                        self.process_plate_recognition(cropped_save_path)
                    except Exception as e:
                        print(f"裁切第一条线图片失败: {str(e)}")

                # 处理穿越第二条线的照片保存
                if crossed_line2:
                    frame_save_count += 1
                    # 裁切图片（第二条线）
                    try:
                        # 将OpenCV图像转换为PIL图像
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        # 裁切图片（第二条线）
                        cropped = pil_image.crop(self.crop_box_second)
                        # 保存裁切后的图片
                        cropped_save_path = os.path.join(video_save_dir, f"{frame_save_count}_line2.jpg")
                        cropped.save(cropped_save_path)
                        # 发送裁切后的图片路径信号
                        self.saved_image_signal.emit(cropped_save_path)
                        # 对裁切后的图片进行车牌识别
                        self.process_plate_recognition(cropped_save_path)
                    except Exception as e:
                        print(f"裁切第二条线图片失败: {str(e)}")

                # 处理穿越第三条线的照片保存
                if crossed_line3:
                    frame_save_count += 1
                    # 裁切图片（第三条线）
                    try:
                        # 将OpenCV图像转换为PIL图像
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        # 裁切图片（第三条线）
                        cropped = pil_image.crop(self.crop_box_third)
                        # 保存裁切后的图片
                        cropped_save_path = os.path.join(video_save_dir, f"{frame_save_count}_line3.jpg")
                        cropped.save(cropped_save_path)
                        # 发送裁切后的图片路径信号
                        self.saved_image_signal.emit(cropped_save_path)
                        # 对裁切后的图片进行车牌识别
                        self.process_plate_recognition(cropped_save_path)
                    except Exception as e:
                        print(f"裁切第三条线图片失败: {str(e)}")

                # 处理穿越第四条线的照片保存
                if crossed_line4:
                    frame_save_count += 1
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        cropped = pil_image.crop(self.crop_box_fourth)
                        cropped_save_path = os.path.join(video_save_dir, f"{frame_save_count}_line4.jpg")
                        cropped.save(cropped_save_path)
                        self.saved_image_signal.emit(cropped_save_path)
                        self.process_plate_recognition(cropped_save_path)
                    except Exception as e:
                        print(f"裁切第四条线图片失败: {str(e)}")

                # 处理穿越第五条线的照片保存
                if crossed_line5:
                    frame_save_count += 1
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        cropped = pil_image.crop(self.crop_box_fifth)
                        cropped_save_path = os.path.join(video_save_dir, f"{frame_save_count}_line5.jpg")
                        cropped.save(cropped_save_path)
                        self.saved_image_signal.emit(cropped_save_path)
                        self.process_plate_recognition(cropped_save_path)
                    except Exception as e:
                        print(f"裁切第五条线图片失败: {str(e)}")

                cv2.putText(annotated_frame, f"L→R: {count_left_to_right}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 0, 0), 2)

                # 发送帧到UI显示
                self.change_pixmap_signal.emit(annotated_frame)
                self.count_signal.emit(count_right_to_left + count_left_to_right)

                end_time = time.time()
                frame_time = end_time - start_time
                frame_times.append(frame_time)
                if len(frame_times) >= fps_update_interval:
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    fps_val = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                    self.fps_signal.emit(fps_val)
                    frame_times = frame_times[-fps_update_interval:]

            cap.release()
            print(f"视频处理完成: {video_path}")
            
        except Exception as e:
            print(f"处理视频时出错: {str(e)}")

    def process_plate_recognition(self, image_path):
        """处理车牌识别"""
        if self.plate_model is None:
            print("车牌识别模型未加载")
            return

        try:
            # 获取图片所在文件夹
            folder_path = os.path.dirname(image_path)
            csv_path = os.path.join(folder_path, "plate_results.csv")

            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                print(f"无法读取图片: {image_path}")
                return

            results = self.plate_model(img, imgsz=640, conf=0.5, iou=0.5)

            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                all_boxes = []
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    area = (x2 - x1) * (y2 - y1)
                    if area >= 400:
                        all_boxes.append((x1, y1, x2, y2, cls, conf))

                c_boxes = [b for b in all_boxes if b[4] == 11]
                if not c_boxes:
                    continue

                cx1, cy1, cx2, cy2, _, _ = c_boxes[0]
                c_center_x = (cx1 + cx2) / 2
                c_center_y = (cy1 + cy2) / 2

                # 修改：不再过滤C字符左侧的字符，让所有字符都参与排序
                filtered_boxes = []
                for x1, y1, x2, y2, cls, conf in all_boxes:
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    # 移除位置限制，包含所有字符
                    filtered_boxes.append((x1, y1, x2, y2, cls, conf, center_x, center_y))

                split_line = c_center_y + 60
                first_row = [b for b in filtered_boxes if b[7] < split_line]
                second_row = [b for b in filtered_boxes if b[7] >= split_line]

                # 按x坐标排序（从左到右）
                first_row.sort(key=lambda b: b[6])
                second_row.sort(key=lambda b: b[6])

                # 组合两行：第一行 + 第二行
                ordered_boxes = first_row + second_row

                char_str = "".join([result.names[b[4]] for b in ordered_boxes])

                # 保存到CSV
                with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if os.path.getsize(csv_path) == 0:  # 如果文件为空，写入表头
                        writer.writerow(["image_name", "plate_number", "timestamp"])
                    writer.writerow([os.path.basename(image_path), char_str,
                                     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                print(f"车牌识别结果: {char_str} - {image_path}")

        except Exception as e:
            print(f"车牌识别处理出错: {str(e)}")

    def stop(self):
        self.running = False
        self.auto_processing = False
        self.wait()


# FullscreenWindow 类已删除


# 修改视频文件夹选择对话框
class VideoFolderSelector(QDialog):
    def __init__(self, parent=None, output_dir="", current_video_name=""):
        super().__init__(parent)
        self.setWindowTitle("选择视频结果")
        self.resize(500, 400)
        self.output_dir = output_dir
        self.current_video_name = current_video_name
        self.selected_folder = ""

        layout = QVBoxLayout(self)

        # 添加说明标签
        info_label = QLabel("请选择要查看的视频结果文件夹:")
        layout.addWidget(info_label)

        # 文件夹列表
        self.folder_list = QListWidget()
        self.folder_list.setSelectionMode(QListWidget.ExtendedSelection)  # 允许多选
        self.folder_list.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.folder_list)

        # 添加删除按钮组
        delete_layout = QHBoxLayout()

        self.delete_selected_btn = QPushButton("删除选中文件夹")
        self.delete_selected_btn.setIcon(QIcon.fromTheme("edit-delete"))
        self.delete_selected_btn.clicked.connect(self.delete_selected_folders)
        delete_layout.addWidget(self.delete_selected_btn)

        self.delete_all_btn = QPushButton("删除所有文件夹")
        self.delete_all_btn.setIcon(QIcon.fromTheme("edit-delete"))
        self.delete_all_btn.clicked.connect(self.delete_all_folders)
        delete_layout.addWidget(self.delete_all_btn)

        self.refresh_btn = QPushButton("刷新列表")
        self.refresh_btn.clicked.connect(self.load_folders)
        delete_layout.addWidget(self.refresh_btn)

        layout.addLayout(delete_layout)

        # 按钮
        button_layout = QHBoxLayout()
        self.open_btn = QPushButton("打开")
        self.open_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.open_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)

        # 加载文件夹
        self.load_folders()

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

    def load_folders(self):
        self.folder_list.clear()
        if not os.path.exists(self.output_dir):
            return

        # 查找所有子文件夹
        for item in os.listdir(self.output_dir):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path):
                list_item = QListWidgetItem(item)
                self.folder_list.addItem(list_item)

                # 如果是当前视频名，设置高亮
                if item == self.current_video_name:
                    list_item.setBackground(QBrush(QColor(200, 230, 255)))
                    self.folder_list.setCurrentItem(list_item)

    def get_selected_folder(self):
        selected_items = self.folder_list.selectedItems()
        if selected_items:
            self.selected_folder = selected_items[0].text()
        return self.selected_folder

    def delete_selected_folders(self):
        """删除选中的文件夹"""
        selected_items = self.folder_list.selectedItems()
        if not selected_items:
            return

        # 确认删除
        count = len(selected_items)
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除选中的 {count} 个文件夹吗？此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # 执行删除
        deleted_count = 0
        error_count = 0

        for item in selected_items:
            folder_name = item.text()
            folder_path = os.path.join(self.output_dir, folder_name)
            try:
                shutil.rmtree(folder_path)
                self.folder_list.takeItem(self.folder_list.row(item))
                deleted_count += 1
            except Exception as e:
                error_count += 1
                print(f"删除文件夹失败: {folder_path}, 错误: {str(e)}")

        # 显示结果
        if error_count > 0:
            QMessageBox.warning(self, "删除结果", f"成功删除 {deleted_count} 个文件夹，{error_count} 个文件夹删除失败。")
        else:
            QMessageBox.information(self, "删除结果", f"已成功删除 {deleted_count} 个文件夹。")

    def delete_all_folders(self):
        """删除所有文件夹"""
        if self.folder_list.count() == 0:
            return

        # 确认删除
        count = self.folder_list.count()
        reply = QMessageBox.question(
            self,
            "确认删除全部",
            f"确定要删除所有 {count} 个文件夹吗？此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # 执行删除
        deleted_count = 0
        error_count = 0

        for i in range(self.folder_list.count() - 1, -1, -1):
            item = self.folder_list.item(i)
            folder_name = item.text()
            folder_path = os.path.join(self.output_dir, folder_name)
            try:
                shutil.rmtree(folder_path)
                self.folder_list.takeItem(i)
                deleted_count += 1
            except Exception as e:
                error_count += 1
                print(f"删除文件夹失败: {folder_path}, 错误: {str(e)}")

        # 显示结果
        if error_count > 0:
            QMessageBox.warning(self, "删除结果", f"成功删除 {deleted_count} 个文件夹，{error_count} 个文件夹删除失败。")
        else:
            QMessageBox.information(self, "删除结果", f"已成功删除所有 {deleted_count} 个文件夹。")


# 修改结果查看对话框
class ResultViewerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("查看结果")
        self.resize(1200, 800)

        main_layout = QVBoxLayout(self)

        # 添加搜索和工具栏
        toolbar_layout = QHBoxLayout()

        # 添加按时间查找功能
        date_group = QGroupBox("按时间筛选")
        date_layout = QHBoxLayout()

        self.filter_enabled = QCheckBox("启用时间筛选")
        date_layout.addWidget(self.filter_enabled)

        self.date_from = QDateTimeEdit()
        self.date_from.setDateTime(QDateTime.currentDateTime().addDays(-7))  # 默认过去7天
        self.date_from.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        date_layout.addWidget(QLabel("从:"))
        date_layout.addWidget(self.date_from)

        self.date_to = QDateTimeEdit()
        self.date_to.setDateTime(QDateTime.currentDateTime())  # 默认到当前时间
        self.date_to.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        date_layout.addWidget(QLabel("到:"))
        date_layout.addWidget(self.date_to)

        self.apply_filter_btn = QPushButton("应用筛选")
        self.apply_filter_btn.clicked.connect(self.apply_time_filter)
        date_layout.addWidget(self.apply_filter_btn)

        date_group.setLayout(date_layout)
        toolbar_layout.addWidget(date_group)

        # 添加删除按钮组
        delete_group = QGroupBox("数据管理")
        delete_layout = QHBoxLayout()

        self.delete_btn = QPushButton("删除选中")
        self.delete_btn.setIcon(QIcon.fromTheme("edit-delete"))
        self.delete_btn.clicked.connect(self.delete_selected)
        delete_layout.addWidget(self.delete_btn)

        self.delete_all_btn = QPushButton("删除全部")
        self.delete_all_btn.setIcon(QIcon.fromTheme("edit-delete"))
        self.delete_all_btn.clicked.connect(self.delete_all)
        delete_layout.addWidget(self.delete_all_btn)

        self.refresh_btn = QPushButton("刷新列表")
        self.refresh_btn.clicked.connect(self.refresh_file_list)
        delete_layout.addWidget(self.refresh_btn)

        delete_group.setLayout(delete_layout)
        toolbar_layout.addWidget(delete_group)

        main_layout.addLayout(toolbar_layout)

        # 创建文件搜索
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("搜索文件:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入关键字搜索文件...")
        self.search_input.textChanged.connect(self.filter_files)
        search_layout.addWidget(self.search_input)
        main_layout.addLayout(search_layout)

        # 创建一个分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, 1)  # 让分割器占用剩余的所有空间

        # 左侧文件列表
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)  # 允许多选
        self.file_list.currentRowChanged.connect(self.display_image)
        self.file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.show_context_menu)
        left_layout.addWidget(self.file_list)

        # 文件计数标签
        self.file_count_label = QLabel("共 0 个文件")
        left_layout.addWidget(self.file_count_label)

        splitter.addWidget(left_widget)

        # 右侧图像显示
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        scroll_area.setWidget(self.image_label)
        splitter.addWidget(scroll_area)

        # 设置分割比例
        splitter.setSizes([300, 900])

        # 存储当前目录和图片路径
        self.current_directory = ""
        self.image_paths = []
        self.all_image_paths = []  # 存储未经筛选的所有图片路径

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

    def show_context_menu(self, position):
        """显示右键菜单"""
        # 创建右键菜单
        menu = QMenu()
        delete_action = menu.addAction("删除选中")

        # 如果选中了多个项目，显示批量删除选项
        if len(self.file_list.selectedItems()) > 1:
            delete_action.setText(f"删除选中的 {len(self.file_list.selectedItems())} 项")

        # 显示菜单并获取用户的选择
        action = menu.exec_(self.file_list.mapToGlobal(position))

        # 处理用户的选择
        if action == delete_action:
            self.delete_selected()

    def load_images_from_directory(self, directory):
        """加载给定目录中的所有图像"""
        self.current_directory = directory

        if os.path.exists(directory):
            self.file_list.clear()
            self.image_paths = []
            self.all_image_paths = []

            # 获取所有jpg图片
            files = []
            for file in os.listdir(directory):
                if file.lower().endswith('.jpg'):
                    file_path = os.path.join(directory, file)
                    files.append((file, file_path))
                    self.all_image_paths.append(file_path)

            # 按照帧号数字排序
            def get_frame_number(filename_tuple):
                # 从 frame_X.jpg 格式中提取数字X
                try:
                    # 假设文件名格式为 frame_123.jpg
                    return int(filename_tuple[0].split('_')[1].split('.')[0])
                except (IndexError, ValueError):
                    # 如果格式不匹配，返回0
                    return 0

            # 按照帧号排序
            files.sort(key=get_frame_number)

            # 添加排序后的文件到列表
            for file, file_path in files:
                self.image_paths.append(file_path)
                item = QListWidgetItem(file)
                item.setData(Qt.UserRole, file_path)  # 存储文件完整路径
                self.file_list.addItem(item)

            # 如果有图片，显示第一张
            if self.image_paths:
                self.file_list.setCurrentRow(0)

            # 更新文件计数
            self.update_file_count()

    def update_file_count(self):
        """更新文件计数标签"""
        count = self.file_list.count()
        total = len(self.all_image_paths)

        if count == total:
            self.file_count_label.setText(f"共 {count} 个文件")
        else:
            self.file_count_label.setText(f"显示 {count}/{total} 个文件")

    def display_image(self, index):
        """显示选中的图像"""
        if 0 <= index < len(self.image_paths):
            pixmap = QPixmap(self.image_paths[index])
            # 按照窗口大小缩放图片，保持比例
            pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(),
                                   Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)

            # 显示当前图像的信息
            file_info = QFileInfo(self.image_paths[index])
            created_time = file_info.birthTime().toString("yyyy-MM-dd hh:mm:ss")
            modified_time = file_info.lastModified().toString("yyyy-MM-dd hh:mm:ss")
            file_size = file_info.size() / 1024  # KB

            info_text = f"文件: {file_info.fileName()} | 创建时间: {created_time} | 修改时间: {modified_time} | 大小: {file_size:.1f} KB"
            self.setWindowTitle(f"查看结果 - {info_text}")

    def delete_selected(self):
        """删除选中的图像文件"""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return

        # 确认删除
        count = len(selected_items)
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除选中的 {count} 个文件吗？此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # 执行删除
        deleted_count = 0
        error_count = 0
        for item in selected_items:
            file_path = item.data(Qt.UserRole)
            try:
                os.remove(file_path)
                # 从列表和图片路径中移除
                row = self.file_list.row(item)
                self.file_list.takeItem(row)
                if file_path in self.image_paths:
                    self.image_paths.remove(file_path)
                if file_path in self.all_image_paths:
                    self.all_image_paths.remove(file_path)
                deleted_count += 1
            except Exception as e:
                error_count += 1
                print(f"删除文件失败: {file_path}, 错误: {str(e)}")

        # 更新文件计数
        self.update_file_count()

        # 显示结果
        if error_count > 0:
            QMessageBox.warning(self, "删除结果", f"成功删除 {deleted_count} 个文件，{error_count} 个文件删除失败。")
        else:
            QMessageBox.information(self, "删除结果", f"已成功删除 {deleted_count} 个文件。")

        # 如果列表不为空，选择第一项
        if self.file_list.count() > 0:
            self.file_list.setCurrentRow(0)
        else:
            # 清空图像显示
            self.image_label.clear()
            self.setWindowTitle("查看结果")

    def delete_all(self):
        """删除当前目录下的所有图像文件"""
        if not self.image_paths:
            return

        # 确认删除
        count = len(self.all_image_paths)
        reply = QMessageBox.question(
            self,
            "确认删除全部",
            f"确定要删除当前目录下的所有 {count} 个文件吗？此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # 执行删除
        deleted_count = 0
        error_count = 0

        for file_path in self.all_image_paths[:]:  # 使用副本进行迭代
            try:
                os.remove(file_path)
                self.image_paths = [p for p in self.image_paths if p != file_path]
                self.all_image_paths.remove(file_path)
                deleted_count += 1
            except Exception as e:
                error_count += 1
                print(f"删除文件失败: {file_path}, 错误: {str(e)}")

        # 清空列表
        self.file_list.clear()

        # 更新文件计数
        self.update_file_count()

        # 清空图像显示
        self.image_label.clear()
        self.setWindowTitle("查看结果")

        # 显示结果
        if error_count > 0:
            QMessageBox.warning(self, "删除结果", f"成功删除 {deleted_count} 个文件，{error_count} 个文件删除失败。")
        else:
            QMessageBox.information(self, "删除结果", f"已成功删除目录下的所有 {deleted_count} 个文件。")

    def apply_time_filter(self):
        """应用时间筛选"""
        if not self.filter_enabled.isChecked() or not self.all_image_paths:
            # 如果未启用筛选或没有图片，显示所有图片
            self.refresh_file_list()
            return

        # 获取时间范围
        from_time = self.date_from.dateTime().toSecsSinceEpoch()
        to_time = self.date_to.dateTime().toSecsSinceEpoch()

        # 清空当前列表
        self.file_list.clear()
        self.image_paths = []

        # 按时间筛选文件
        for file_path in self.all_image_paths:
            file_info = QFileInfo(file_path)
            file_time = file_info.lastModified().toSecsSinceEpoch()

            if from_time <= file_time <= to_time:
                self.image_paths.append(file_path)
                file_name = os.path.basename(file_path)
                item = QListWidgetItem(file_name)
                item.setData(Qt.UserRole, file_path)
                self.file_list.addItem(item)

        # 按照帧号排序列表项
        self.sort_file_list()

        # 更新文件计数
        self.update_file_count()

        # 如果有图片，选择第一张
        if self.image_paths:
            self.file_list.setCurrentRow(0)

    def filter_files(self, text):
        """根据文本筛选文件名"""
        if not text:
            # 如果搜索框为空，恢复时间筛选或显示所有
            self.apply_time_filter()
            return

        # 先应用时间筛选获取基础列表
        if self.filter_enabled.isChecked():
            # 获取时间范围内的文件
            self.apply_time_filter()
            # 获取当前列表中的所有文件路径
            paths_to_filter = [self.file_list.item(i).data(Qt.UserRole)
                               for i in range(self.file_list.count())]
        else:
            # 使用所有文件
            paths_to_filter = self.all_image_paths

        # 清空当前列表
        self.file_list.clear()
        self.image_paths = []

        # 根据文本筛选
        for file_path in paths_to_filter:
            file_name = os.path.basename(file_path)
            if text.lower() in file_name.lower():
                self.image_paths.append(file_path)
                item = QListWidgetItem(file_name)
                item.setData(Qt.UserRole, file_path)
                self.file_list.addItem(item)

        # 按照帧号排序列表项
        self.sort_file_list()

        # 更新文件计数
        self.update_file_count()

        # 如果有图片，选择第一张
        if self.image_paths:
            self.file_list.setCurrentRow(0)

    def sort_file_list(self):
        """按照帧号排序文件列表"""
        # 提取所有项和它们的排序键
        items_with_keys = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            file_name = item.text()
            try:
                # 假设文件名格式为 frame_123.jpg
                frame_number = int(file_name.split('_')[1].split('.')[0])
            except (IndexError, ValueError):
                frame_number = 0
            items_with_keys.append((item, frame_number))

        # 按排序键排序
        items_with_keys.sort(key=lambda x: x[1])

        # 重新构建图片路径和列表
        self.file_list.clear()
        self.image_paths = []

        for item, _ in items_with_keys:
            file_path = item.data(Qt.UserRole)
            self.image_paths.append(file_path)
            self.file_list.addItem(QListWidgetItem(item))

    def refresh_file_list(self):
        """刷新文件列表"""
        if self.current_directory:
            self.load_images_from_directory(self.current_directory)


class WaterCarriageViewer(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("积水车厢查看与审核")
        self.resize(1100, 700)
        self.filter_days = 7
        self.parent_window = parent  # 保存父窗口引用，用于通知刷新

        layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(6, 6, 6, 6)
        toolbar.setSpacing(10)
        header = QLabel("人工审核")
        header.setStyleSheet("font-weight:600;color:#333;")
        toolbar.addWidget(header)

        toolbar.addSpacing(12)
        toolbar.addWidget(QLabel("时间范围:"))
        self.days_combo = QComboBox()
        self.days_combo.addItems(["最近7天", "最近30天", "最近90天", "全部"])
        self.days_combo.setCurrentIndex(0)
        self.days_combo.currentIndexChanged.connect(self.on_days_changed)
        toolbar.addWidget(self.days_combo)

        self.refresh_btn_wc = QPushButton("刷新")
        self.mark_no_water_btn = QPushButton("标记为无水")
        self.confirm_water_btn = QPushButton("确认有水")
        toolbar.addWidget(self.refresh_btn_wc)
        toolbar.addStretch(1)
        self.stats_label = QLabel("共 0 条记录")
        self.stats_label.setStyleSheet("color:#666;")
        toolbar.addWidget(self.stats_label)
        toolbar.addWidget(self.mark_no_water_btn)
        toolbar.addWidget(self.confirm_water_btn)
        layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)

        # 左侧表格
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "序号", "列车名", "车厢号", "完整车号", "积水占比(%)", "检测时间"
        ])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self.on_table_selection_changed)
        left_layout.addWidget(self.table)
        splitter.addWidget(left_panel)

        # 右侧图片预览
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(12)

        top_label = QLabel("顶部积水照片")
        top_label.setStyleSheet("font-weight:600;")
        right_layout.addWidget(top_label)
        self.top_image_scroll = QScrollArea()
        self.top_image_scroll.setWidgetResizable(True)
        self.top_image_label = QLabel("无图片")
        self.top_image_label.setAlignment(Qt.AlignCenter)
        self.top_image_label.setStyleSheet("color:#888; border:1px dashed #ccc; padding:20px;")
        self.top_image_scroll.setWidget(self.top_image_label)
        right_layout.addWidget(self.top_image_scroll, 1)

        side_label = QLabel("侧面照片")
        side_label.setStyleSheet("font-weight:600;")
        right_layout.addWidget(side_label)
        self.side_image_scroll = QScrollArea()
        self.side_image_scroll.setWidgetResizable(True)
        self.side_image_label = QLabel("无图片")
        self.side_image_label.setAlignment(Qt.AlignCenter)
        self.side_image_label.setStyleSheet("color:#888; border:1px dashed #ccc; padding:20px;")
        self.side_image_scroll.setWidget(self.side_image_label)
        right_layout.addWidget(self.side_image_scroll, 1)

        splitter.addWidget(right_panel)
        splitter.setSizes([650, 450])
        layout.addWidget(splitter, 1)

        self.refresh_btn_wc.clicked.connect(self.refresh_data)
        self.mark_no_water_btn.clicked.connect(self.batch_mark_no_water)
        self.confirm_water_btn.clicked.connect(self.batch_confirm_water)

        self.load_data()

    def on_days_changed(self, index):
        days_map = {0: 7, 1: 30, 2: 90, 3: 0}
        self.filter_days = days_map.get(index, 7)
        self.load_data()

    def refresh_data(self):
        self.load_data()

    def load_data(self):
        if query_recent_water_carriages is None:
            QMessageBox.warning(self, "警告", "查询函数不可用")
            return
        try:
            rows = query_recent_water_carriages(days=self.filter_days, limit=500)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"查询失败: {str(e)}")
            return

        self.table.setRowCount(0)
        for row_idx, r in enumerate(rows):
            self.table.insertRow(row_idx)
            data_map = {
                'train_name': r.get('train_name', ''),
                'carriage_id': str(r.get('carriage_id', '')),
                'full_carriage_number': r.get('full_carriage_number', ''),
                'top_image_path': r.get('top_image_path', ''),
                'side_image_path': r.get('side_image_path', ''),
            }
            values = [
                str(row_idx + 1),
                data_map['train_name'],
                data_map['carriage_id'],
                r.get('full_carriage_number', ''),
                f"{float(r.get('water_area_ratio', 0.0)):.2f}%",
                r.get('recognition_time', ''),
            ]
            for c, v in enumerate(values):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                if c == 1:
                    item.setData(Qt.UserRole, data_map)
                self.table.setItem(row_idx, c, item)

        self.stats_label.setText(f"共 {self.table.rowCount()} 条记录")

        if self.table.rowCount() > 0:
            self.table.selectRow(0)
        else:
            self.clear_top_image()
            self.clear_side_image()

    def selected_items(self):
        sel = []
        for idx in self.table.selectionModel().selectedRows():
            train_item = self.table.item(idx.row(), 1)
            carriage_item = self.table.item(idx.row(), 2)
            full_num_item = self.table.item(idx.row(), 3)

            train_name = train_item.text() if train_item else ""
            carriage_id = carriage_item.text() if carriage_item else ""
            full_number = full_num_item.text() if full_num_item else ""
            meta = train_item.data(Qt.UserRole) if train_item else {}
            if meta:
                full_number = meta.get('full_carriage_number', full_number)
            sel.append({
                "train_name": train_name,
                "carriage_id": carriage_id,
                "full_carriage_number": full_number,
            })
        return sel

    def on_table_selection_changed(self):
        indexes = self.table.selectionModel().selectedRows()
        if not indexes:
            self.clear_top_image()
            self.clear_side_image()
            return
        row = indexes[0].row()
        meta_item = self.table.item(row, 1)
        meta = meta_item.data(Qt.UserRole) if meta_item else None
        self.show_preview(meta)

    def show_preview(self, meta):
        if not meta:
            self.clear_top_image()
            self.clear_side_image()
            return

        train_name = meta.get('train_name')
        carriage_id = meta.get('carriage_id')

        top_path = self._resolve_image_path(meta.get('top_image_path'))
        if not self.display_image(top_path, self.top_image_label):
            fallback = self._build_fallback_path(train_name, carriage_id)
            if not self.display_image(fallback, self.top_image_label):
                self.clear_top_image()

        side_path = self._resolve_image_path(meta.get('side_image_path'))
        if not self.display_image(side_path, self.side_image_label):
            self.clear_side_image()

    def display_image(self, path, target_label):
        if not path or not os.path.exists(path) or not os.path.isfile(path):
            return False
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return False
        target_label.setPixmap(pixmap.scaled(
            target_label.width() or pixmap.width(),
            target_label.height() or pixmap.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        target_label.setStyleSheet("")
        return True

    def clear_top_image(self):
        self.top_image_label.setPixmap(QPixmap())
        self.top_image_label.setText("无图片")
        self.top_image_label.setStyleSheet("color:#888; border:1px dashed #ccc; padding:20px;")

    def clear_side_image(self):
        self.side_image_label.setPixmap(QPixmap())
        self.side_image_label.setText("无图片")
        self.side_image_label.setStyleSheet("color:#888; border:1px dashed #ccc; padding:20px;")

    def _resolve_image_path(self, path):
        if not path:
            return ""
        return path if os.path.isabs(path) else os.path.abspath(path)

    def _build_fallback_path(self, train_name, carriage_id):
        try:
            carriage_num = int(carriage_id) if carriage_id and str(carriage_id).isdigit() else None
        except Exception:
            carriage_num = None
        if not train_name or carriage_num is None:
            return ""
        frame_name = f"frame_{carriage_num}"
        return os.path.join(
            OUTPUT_IMAGES_DIR, train_name, "积水识别结果",
            f"{frame_name}_积水识别.png"
        )

    def batch_mark_no_water(self):
        items = self.selected_items()
        if not items:
            QMessageBox.information(self, "提示", "请先选择记录")
            return
        if review_mark_no_water is None:
            QMessageBox.warning(self, "警告", "审核函数不可用")
            return
        try:
            result = review_mark_no_water(items)
            updated = len(result.get('updated', []))
            failed = result.get('failed', [])
            
            # 同步更新CSV文件
            self.update_csv_water_status(items, has_water=False)
            
            # 清除相关缓存
            self.clear_related_cache(items)
            
            # 通知列车信息查看窗口刷新
            self.notify_train_info_refresh(items)
            
            msg = f"成功更新 {updated} 条。"
            if failed:
                msg += f"\n失败 {len(failed)} 条。"
            QMessageBox.information(self, "结果", msg)
            self.load_data()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新失败: {str(e)}")

    def batch_confirm_water(self):
        items = self.selected_items()
        if not items:
            QMessageBox.information(self, "提示", "请先选择记录")
            return
        if review_confirm_water is None:
            QMessageBox.warning(self, "警告", "审核函数不可用")
            return
        try:
            result = review_confirm_water(items)
            updated = len(result.get('updated', []))
            failed = result.get('failed', [])
            
            # 同步更新CSV文件
            self.update_csv_water_status(items, has_water=True)
            
            # 清除相关缓存
            self.clear_related_cache(items)
            
            # 通知列车信息查看窗口刷新
            self.notify_train_info_refresh(items)
            
            msg = f"成功更新 {updated} 条。"
            if failed:
                msg += f"\n失败 {len(failed)} 条。"
            QMessageBox.information(self, "结果", msg)
            self.load_data()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新失败: {str(e)}")
    
    def update_csv_water_status(self, items, has_water):
        """更新CSV文件中的积水状态"""
        try:
            # 按列车名分组处理
            train_groups = {}
            for item in items:
                train_name = item.get('train_name', '')
                carriage_id = item.get('carriage_id', '')
                if train_name and carriage_id:
                    if train_name not in train_groups:
                        train_groups[train_name] = []
                    train_groups[train_name].append(carriage_id)
            
            # 处理每个列车的CSV文件
            summary_csv = os.path.join(RUNS_OUTPUT_IMAGES_DIR, "所有子文件夹积水识别结果汇总.csv")
            summary_updated = False
            summary_rows = []
            summary_fieldnames = None
            
            # 如果需要更新汇总文件，先读取它
            if os.path.exists(summary_csv):
                try:
                    with open(summary_csv, 'r', encoding='utf-8-sig') as f:
                        reader = csv.DictReader(f)
                        summary_fieldnames = reader.fieldnames
                        summary_rows = list(reader)
                except Exception as e:
                    print(f"读取汇总CSV文件失败: {str(e)}")
            
            for train_name, carriage_ids in train_groups.items():
                # 优先查找个别列车的CSV文件
                individual_csv = os.path.join(RUNS_OUTPUT_IMAGES_DIR, train_name, f"{train_name}_积水识别结果.csv")
                
                # 更新个别CSV文件（如果存在）
                if os.path.exists(individual_csv):
                    self._update_single_csv(individual_csv, train_name, carriage_ids, has_water, is_summary=False)
                
                # 同时更新汇总CSV文件（如果存在）
                if summary_rows and summary_fieldnames:
                    summary_updated = self._update_single_csv_rows(
                        summary_rows, train_name, carriage_ids, has_water, is_summary=True
                    ) or summary_updated
            
            # 写回汇总CSV文件（如果有更新）
            if summary_updated and summary_fieldnames and summary_rows:
                try:
                    with open(summary_csv, 'w', newline='', encoding='utf-8-sig') as f:
                        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
                        writer.writeheader()
                        writer.writerows(summary_rows)
                    print(f"已更新汇总CSV文件 {summary_csv}")
                except Exception as e:
                    print(f"写入汇总CSV文件失败: {str(e)}")
                    
        except Exception as e:
            print(f"更新CSV文件时出错: {str(e)}")
    
    def _update_single_csv(self, csv_path, train_name, carriage_ids, has_water, is_summary=False):
        """更新单个CSV文件"""
        try:
            # 读取CSV文件
            rows = []
            fieldnames = None
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                rows = list(reader)
            
            # 更新记录
            updated_count = self._update_single_csv_rows(rows, train_name, carriage_ids, has_water, is_summary)
            
            # 写回CSV文件
            if updated_count > 0 and fieldnames:
                with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"已更新CSV文件 {csv_path}，修改了 {updated_count} 条记录")
                return True
            return False
        except Exception as e:
            print(f"更新CSV文件 {csv_path} 时出错: {str(e)}")
            return False
    
    def _update_single_csv_rows(self, rows, train_name, carriage_ids, has_water, is_summary=False):
        """更新CSV行数据"""
        updated_count = 0
        
        for row in rows:
            # 对于汇总文件，需要检查子文件夹是否匹配
            if is_summary:
                subfolder = row.get('子文件夹', '').strip()
                if subfolder != train_name:
                    continue
            
            # 获取图片名称
            image_name = row.get('原图片名称', '').strip()
            if not image_name:
                continue
            
            # 从图片名称中提取车厢号（frame_X.jpg -> X）
            try:
                frame_num = int(image_name.replace('frame_', '').replace('.jpg', ''))
                
                # 检查是否匹配任意一个车厢ID
                matched = False
                for carriage_id in carriage_ids:
                    carriage_num = int(carriage_id)
                    if frame_num == carriage_num:
                        matched = True
                        break
                
                if matched:
                    # 更新积水状态
                    row['是否有积水'] = '是' if has_water else '否'
                    
                    # 如果标记为无水，清空识别后图片路径和积水面积占比
                    if not has_water:
                        row['识别后图片路径'] = ''
                        row['积水面积占比(%)'] = '0.0'
                    
                    updated_count += 1
            except (ValueError, AttributeError):
                continue
        
        return updated_count
    
    def clear_related_cache(self, items):
        """清除相关缓存"""
        try:
            # 获取所有涉及的列车名
            train_names = set()
            for item in items:
                train_name = item.get('train_name', '')
                if train_name:
                    train_names.add(train_name)
            
            # 如果父窗口是MainWindow，尝试清除缓存
            if self.parent_window and hasattr(self.parent_window, 'train_info_window'):
                train_window = self.parent_window.train_info_window
                if train_window and hasattr(train_window, 'cache_manager'):
                    # 清除列车数据缓存
                    for train_name in train_names:
                        train_window.cache_manager.invalidate_train_cache(train_name)
                    
                    # 清除CSV缓存
                    csv_keys_to_remove = []
                    for csv_path in train_window.cache_manager.csv_cache.keys():
                        # 检查是否与修改的列车相关
                        for train_name in train_names:
                            if train_name in csv_path:
                                csv_keys_to_remove.append(csv_path)
                                break
                    
                    for csv_path in csv_keys_to_remove:
                        if csv_path in train_window.cache_manager.csv_cache:
                            del train_window.cache_manager.csv_cache[csv_path]
                    
                    if train_names:
                        print(f"已清除列车 {train_names} 的缓存")
        except Exception as e:
            print(f"清除缓存时出错: {str(e)}")
    
    def notify_train_info_refresh(self, items):
        """通知列车信息查看窗口刷新"""
        try:
            if self.parent_window and hasattr(self.parent_window, 'train_info_window'):
                train_window = self.parent_window.train_info_window
                if train_window and train_window.isVisible():
                    # 获取当前选中的列车名
                    current_train = None
                    if hasattr(train_window, 'train_combo'):
                        current_train = train_window.train_combo.currentText()
                    
                    # 检查是否有修改的列车与当前显示的列车匹配
                    modified_trains = {item.get('train_name', '') for item in items}
                    if current_train in modified_trains:
                        # 刷新当前列车数据
                        train_window.on_train_selected(train_window.train_combo.currentIndex())
                        print(f"已刷新列车信息查看窗口: {current_train}")
        except Exception as e:
            print(f"通知刷新时出错: {str(e)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("中国移动-广钢积水检测v5.0")
        self.setMinimumSize(1000, 800)

        # 设置窗口居中显示在桌面上
        self.center_window()

        # 初始化默认路径
        self.model_path = SEGMENT_TRAIN2_WEIGHTS
        self.save_path = RUNS_OUTPUT_IMAGES_DIR
        self.auto_folder_path = r"E:\积水识别项目\视频下载模块\record"  # 设置默认的自动处理文件夹路径

        # 积水识别相关变量
        self.water_model_path = WATER_SEGMENT_WEIGHTS  # 积水识别模型路径
        self.water_root_folder = RUNS_OUTPUT_IMAGES_DIR  # 积水识别根文件夹路径
        self.water_processed_files = PROCESSED_FILES_PATH  # 记录已处理文件的文件
        self.water_check_interval = 30  # 定时扫描间隔（秒）
        self.water_monitor = None  # 积水识别监控器
        self.water_monitor_running = False  # 监控状态标记

        # 设置简洁的主界面样式（去除渐变色，使用纯色背景）
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QWidget {
                font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
                font-size: 13px;
                background-color: #f0f0f0;
            }
            QLabel {
                color: #333333;
            }
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 6px;
                font-weight: 500;
                font-size: 13px;
                min-width: 120px;
                min-height: 16px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2c5f9e;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QGroupBox {
                font-weight: 600;
                font-size: 14px;
                border: 1px solid #cccccc;
                border-radius: 6px;
                margin-top: 18px;
                padding-top: 25px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 15px;
                padding: 8px 15px;
                background-color: #4a90e2;
                color: white;
                border-radius: 4px;
                font-weight: 600;
            }
            QTableWidget {
                border: 1px solid #cccccc;
                border-radius: 6px;
                background-color: #ffffff;
                gridline-color: #eeeeee;
                selection-background-color: #d0e0f0;
            }
            QTableWidget::item {
                padding: 10px;
                border-bottom: 1px solid #eeeeee;
            }
            QTableWidget::item:selected {
                background-color: #d0e0f0;
                color: #333333;
            }
            QHeaderView::section {
                background-color: #4a90e2;
                color: white;
                padding: 12px;
                border: none;
                font-weight: 600;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 14px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical {
                background: #cccccc;
                min-height: 30px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical:hover {
                background: #999999;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QFrame {
                background-color: #f0f0f0;
            }
        """)

        # 创建主窗口部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # 创建设置布局的容器
        settings_container = QWidget()
        settings_container.setObjectName("settingsContainer")
        settings_container.setStyleSheet("""
            #settingsContainer {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #cccccc;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 20px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                color: #4a90e2;
            }
        """)
        settings_layout = QGridLayout(settings_container)
        settings_layout.setContentsMargins(15, 15, 15, 15)
        settings_layout.setSpacing(10)

        # 文件操作组已隐藏

        # 统一按钮样式
        button_style = """
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                font-weight: 500;
                font-size: 12px;
                min-width: 100px;
                min-height: 24px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2c5f9e;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """
        
        # 创建现代化自动处理组
        auto_group = QGroupBox("⚡ 自动处理c")
        auto_layout = QHBoxLayout(auto_group)
        auto_layout.setSpacing(10)
        auto_layout.setContentsMargins(10, 10, 10, 10)

        self.auto_folder_btn = QPushButton("📁 选择自动处理文件夹")
        self.auto_folder_btn.setStyleSheet(button_style)
        self.auto_folder_btn.clicked.connect(self.select_auto_folder)
        auto_layout.addWidget(self.auto_folder_btn)

        self.auto_start_btn = QPushButton("▶️ 开始自动处理d")
        self.auto_start_btn.setStyleSheet(button_style)
        self.auto_start_btn.clicked.connect(self.start_auto_processing)
        auto_layout.addWidget(self.auto_start_btn)

        self.auto_stop_btn = QPushButton("⏹️ 停止自动处理")
        self.auto_stop_btn.setStyleSheet(button_style)
        self.auto_stop_btn.clicked.connect(self.stop_auto_processing)
        auto_layout.addWidget(self.auto_stop_btn)

        # 添加自动处理组
        settings_layout.addWidget(auto_group, 0, 0, 1, 6)

        # 创建现代化结果查看组
        view_group = QGroupBox("📊 结果查看")
        view_layout = QHBoxLayout(view_group)
        view_layout.setSpacing(8)
        view_layout.setContentsMargins(10, 10, 10, 10)
        
        # 统一结果查看按钮样式
        view_button_style = """
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 6px 10px;
                border-radius: 4px;
                font-weight: 500;
                font-size: 11px;
                min-width: 80px;
                min-height: 24px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2c5f9e;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """

        self.train_info_btn = QPushButton("🚆 列车信息查看")
        self.train_info_btn.setStyleSheet(view_button_style)
        self.train_info_btn.clicked.connect(self.open_train_info)
        view_layout.addWidget(self.train_info_btn)

        self.accuracy_viz_btn = QPushButton("📊 准确率可视化")
        self.accuracy_viz_btn.setStyleSheet(view_button_style)
        self.accuracy_viz_btn.clicked.connect(self.open_accuracy_visualization)
        view_layout.addWidget(self.accuracy_viz_btn)
        
        # 新增积水识别系统按钮
        self.water_detection_btn = QPushButton("🌊 积水识别系统")
        self.water_detection_btn.setStyleSheet(view_button_style)
        self.water_detection_btn.clicked.connect(self.open_water_detection)
        view_layout.addWidget(self.water_detection_btn)
        
        # 新增积水车厢查看按钮
        self.water_carriage_btn = QPushButton("💧 积水车厢查看")
        self.water_carriage_btn.setStyleSheet(view_button_style)
        self.water_carriage_btn.clicked.connect(self.open_water_carriage_viewer)
        view_layout.addWidget(self.water_carriage_btn)

        # 添加结果查看组
        settings_layout.addWidget(view_group, 0, 6, 1, 6)

        # ==================== 新增：Cemian系统控制组 ====================
        # 添加Cemian系统侧面控制模块（精简版）
        cemian_group = QGroupBox("🎯 侧面检测系统")
        cemian_layout = QHBoxLayout(cemian_group)
        cemian_layout.setSpacing(10)
        cemian_layout.setContentsMargins(10, 10, 10, 10)

        # 只保留两个核心按钮
        self.cemian_start_auto_btn = QPushButton("🚀 开启自动处理c")
        self.cemian_start_auto_btn.setStyleSheet(button_style)
        self.cemian_start_auto_btn.clicked.connect(self.cemian_start_auto_processing)
        cemian_layout.addWidget(self.cemian_start_auto_btn)

        self.cemian_stop_auto_btn = QPushButton("⏹️ 停止自动处理")
        self.cemian_stop_auto_btn.setStyleSheet(button_style)
        self.cemian_stop_auto_btn.clicked.connect(self.cemian_stop_auto_processing)
        self.cemian_stop_auto_btn.setEnabled(False)
        cemian_layout.addWidget(self.cemian_stop_auto_btn)

        # 添加Cemian系统控制组到布局
        settings_layout.addWidget(cemian_group, 1, 0, 1, 12)  # 占据整行

        cemian_layout.addWidget(self.cemian_stop_auto_btn)

        # 添加Cemian系统控制组到布局
        settings_layout.addWidget(cemian_group, 1, 0, 1, 12)  # 占据整行

        # 添加设置容器到主布局
        layout.addWidget(settings_container, 1)

        # 系统信息卡片已移除
        
        # 添加超级美化的视频预览区域
        video_section = QWidget()
        video_section.setObjectName("videoSection")
        video_section.setStyleSheet("""
            #videoSection {
                background-color: #ffffff;
                border-radius: 10px;
                margin: 15px;
                border: 1px solid #cccccc;
            }
        """)
        video_layout = QVBoxLayout(video_section)
        video_layout.setContentsMargins(15, 10, 15, 10)
        video_layout.setSpacing(10)
        
        # 双视频显示区域（左右分屏）
        video_display_container = QWidget()
        video_display_container.setStyleSheet("""
            background-color: #f8f8f8;
            border-radius: 8px;
            border: 1px solid #cccccc;
        """)
        video_display_layout = QHBoxLayout(video_display_container)  # 改为水平布局
        video_display_layout.setContentsMargins(10, 10, 10, 10)
        video_display_layout.setSpacing(10)
        
        # 左侧：主系统视频
        left_video_section = QWidget()
        left_layout = QVBoxLayout(left_video_section)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # 主系统标题
        main_title = QLabel("🏭 主系统视频")
        main_title.setStyleSheet("""
            font-size: 14px;
            font-weight: 600;
            color: #333333;
            padding: 4px 8px;
            background-color: #f0f0f0;
            border-radius: 4px;
            text-align: center;
        """)
        main_title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(main_title)
        
        # 主系统视频显示
        self.video_label = QLabel()
        self.video_label.setMinimumSize(600, 450)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border: 1px solid #cccccc;
                border-radius: 6px;
                color: #999999;
                font-size: 14px;
                font-weight: 500;
            }
        """)
        self.video_label.setText("📺 等待主系统视频...")
        left_layout.addWidget(self.video_label, 1)
        
        video_display_layout.addWidget(left_video_section)
        
        # 右侧：Cemian系统视频
        right_video_section = QWidget()
        right_layout = QVBoxLayout(right_video_section)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # Cemian系统标题
        cemian_title = QLabel("🎯 侧面检测视频")
        cemian_title.setStyleSheet("""
            font-size: 14px;
            font-weight: 600;
            color: #333333;
            padding: 4px 8px;
            background-color: #f0f0f0;
            border-radius: 4px;
            text-align: center;
        """)
        cemian_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(cemian_title)
        
        # Cemian系统视频显示
        self.cemian_video_label = QLabel()
        self.cemian_video_label.setMinimumSize(600, 450)
        self.cemian_video_label.setAlignment(Qt.AlignCenter)
        self.cemian_video_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border: 1px solid #cccccc;
                border-radius: 6px;
                color: #999999;
                font-size: 14px;
                font-weight: 500;
            }
        """)
        self.cemian_video_label.setText("📺 等待侧面检测视频...")
        right_layout.addWidget(self.cemian_video_label, 1)
        
        video_display_layout.addWidget(right_video_section)
        
        video_layout.addWidget(video_display_container, 1)
        
        # 添加控制面板
        controls_panel = QWidget()
        controls_layout = QHBoxLayout(controls_panel)
        controls_layout.setSpacing(15)
        
        # 统计卡片（只保留处理速度和检测计数）
        stats_cards = [
            ("⚡ 处理速度", "0.0 FPS", "#4a90e2"),
            ("🎯 检测计数", "0", "#4a90e2")
        ]
        
        for title, value, color in stats_cards:
            card = QWidget()
            card.setStyleSheet(f"""
                background-color: #ffffff;
                border-radius: 6px;
                border: 1px solid #cccccc;
            """)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 8, 12, 8)
            
            title_label = QLabel(title)
            title_label.setStyleSheet("""
                color: #333333;
                font-size: 11px;
                font-weight: 600;
            """)
            title_label.setAlignment(Qt.AlignCenter)
            
            value_label = QLabel(value)
            value_label.setStyleSheet(f"""
                color: {color};
                font-size: 16px;
                font-weight: 700;
            """)
            value_label.setAlignment(Qt.AlignCenter)
            
            card_layout.addWidget(title_label)
            card_layout.addWidget(value_label)
            controls_layout.addWidget(card)
        
        # 保存统计标签的引用（只保留处理速度和检测计数）
        self.fps_label = controls_layout.itemAt(0).widget().layout().itemAt(1).widget()
        self.count_label = controls_layout.itemAt(1).widget().layout().itemAt(1).widget()
        
        video_layout.addWidget(controls_panel)
        layout.addWidget(video_section, 4)

        # 移除视频状态表格容器

        # 路径信息容器
        path_container = QWidget()
        path_container.setObjectName("pathContainer")
        path_container.setStyleSheet("""
            #pathContainer {
                background-color: white;
                border-radius: 10px;
                border: 1px solid #e1e1e1;
                padding: 10px;
            }
        """)
        paths_layout = QGridLayout(path_container)

        # 使用图标和样式美化路径显示
        file_icon = QLabel("??")
        file_icon.setStyleSheet("font-size: 16px;")
        self.video_path_label = QLabel("视频文件：未选择")
        self.video_path_label.setStyleSheet("color: #7f8c8d; padding: 3px;")
        paths_layout.addWidget(file_icon, 0, 0)
        paths_layout.addWidget(self.video_path_label, 0, 1)

        model_icon = QLabel("??")
        model_icon.setStyleSheet("font-size: 16px;")
        self.model_path_label = QLabel(f"模型文件：{self.model_path}")
        self.model_path_label.setStyleSheet("color: #7f8c8d; padding: 3px;")
        paths_layout.addWidget(model_icon, 1, 0)
        paths_layout.addWidget(self.model_path_label, 1, 1)

        output_icon = QLabel("??")
        output_icon.setStyleSheet("font-size: 16px;")
        self.output_path_label = QLabel(f"输出目录：{self.save_path}")
        self.output_path_label.setStyleSheet("color: #7f8c8d; padding: 3px;")
        paths_layout.addWidget(output_icon, 2, 0)
        paths_layout.addWidget(self.output_path_label, 2, 1)

        auto_icon = QLabel("??")
        auto_icon.setStyleSheet("font-size: 16px;")
        self.auto_folder_label = QLabel("自动处理文件夹：未选择")
        self.auto_folder_label.setStyleSheet("color: #7f8c8d; padding: 3px;")
        paths_layout.addWidget(auto_icon, 3, 0)
        paths_layout.addWidget(self.auto_folder_label, 3, 1)

        # 路径信息容器已隐藏

        # 移除统计面板

        # 移除视频显示容器

        # 创建视频处理线程（恢复完整功能）
        self.video_thread = VideoThread()
        # 恢复信号连接
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.count_signal.connect(self.update_count)
        self.video_thread.saved_image_signal.connect(self.update_saved_images)
        self.video_thread.fps_signal.connect(self.update_fps)
        self.video_thread.video_finished_signal.connect(self.on_video_processed)

        # 创建自动处理器
        self.auto_processor = AutoProcessor()
        self.auto_processor.status_changed.connect(self.update_video_status)

        self.video_path = ""
        
        # 初始化结果查看器
        self.result_viewer = None

        # 移除全屏窗口相关初始化
        # self.fullscreen_window = None
        # self.is_fullscreen = False

        # 移除结果查看器初始化
        # self.result_viewer = None

        # 保存的图片路径列表
        self.saved_images = []

        # Cemian系统相关变量
        self.cemian_model_path = os.path.normpath(DETECT_TRAIN5_WEIGHTS)
        self.cemian_plate_model_path = os.path.normpath(DETECT_TRAIN2_PLATE_WEIGHTS)
        self.cemian_save_path = OUTPUT_IMAGES_DIR
        self.cemian_video_folder = r"E:\积水识别项目\视频下载模块\record"
        self.cemian_process = None

        # 创建Cemian视频处理线程
        self.cemian_video_thread = CemianVideoThread()
        self.cemian_video_thread.change_pixmap_signal.connect(self.update_cemian_image)
        self.cemian_video_thread.count_signal.connect(self.update_cemian_count)
        self.cemian_video_thread.count_right_to_left_signal.connect(self.update_cemian_count_right_to_left)
        self.cemian_video_thread.count_left_to_right_signal.connect(self.update_cemian_count_left_to_right)
        self.cemian_video_thread.count_line1_signal.connect(self.update_cemian_count_line1)
        self.cemian_video_thread.count_line2_signal.connect(self.update_cemian_count_line2)
        self.cemian_video_thread.saved_image_signal.connect(self.update_cemian_saved_images)
        self.cemian_video_thread.fps_signal.connect(self.update_cemian_fps)

        # 视频状态字典
        self.video_status = {}
        
        # 下载状态追踪（简化版）
        self.download_status = {
            'total_files': 0,
            'completed_files': 0,
            'failed_files': 0,
            'current_downloading': None
        }

        # 设置默认的自动处理文件夹
        if os.path.exists(self.auto_folder_path):
            # 移除标签更新
            # self.auto_folder_label.setText(f"自动处理文件夹：{self.auto_folder_path}")
            self.auto_processor.set_video_folder(self.auto_folder_path)
        
        # 初始化数据库自动更新器（延迟启动，避免阻塞UI初始化）
        self.database_updater = DatabaseAutoUpdater(check_interval=30)
        QTimer.singleShot(0, self.database_updater.start_monitoring)

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

    def select_auto_folder(self):
        """选择自动处理文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择自动处理文件夹")
        if folder:
            self.auto_folder_path = folder
            # 移除标签更新
            # self.auto_folder_label.setText(f"自动处理文件夹：{self.auto_folder_path}")
            self.auto_processor.set_video_folder(folder)

    def start_auto_processing(self):
        """开始自动处理"""
        if not self.auto_folder_path:
            QMessageBox.warning(self, "警告", "请先选择自动处理文件夹")
            return

        # 设置模型和保存路径
        self.video_thread.set_model_path(self.model_path)
        self.video_thread.set_save_path(self.save_path)
        # 移除相关设置
        # self.video_thread.set_conf_threshold(self.conf_spinbox.value())
        # self.video_thread.set_detection_area(self.detection_area_checkbox.isChecked())

        # 启动自动处理器
        self.auto_processor.start()
        
        # 按钮状态与颜色（与侧面检测按钮一致的绿色激活态）
        self.auto_start_btn.setEnabled(False)
        self.auto_stop_btn.setEnabled(True)
        self.auto_start_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;  /* 绿色表示已激活 */
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                font-weight: 500;
                font-size: 12px;
                min-width: 100px;
                min-height: 24px;
            }
            QPushButton:disabled {
                background-color: #28a745;  /* 保持绿色 */
                color: white;
            }
        """)

    def stop_auto_processing(self):
        """停止自动处理"""
        self.auto_processor.stop()
        self.video_thread.stop()
        
        # 恢复按钮原始样式
        button_style = """
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                font-weight: 500;
                font-size: 12px;
                min-width: 100px;
                min-height: 24px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2c5f9e;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """
        self.auto_start_btn.setStyleSheet(button_style)
        self.auto_start_btn.setEnabled(True)
        self.auto_stop_btn.setEnabled(False)

    def update_video_status(self, video_path, status):
        """更新视频状态"""
        video_name = os.path.basename(video_path)

        # 更新状态字典
        self.video_status[video_path] = {
            'status': status,
            'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # 移除表格更新
        # self.update_video_table()

        # 如果是正在处理的视频，开始处理
        if status == "正在处理":
            # 如果当前有视频在处理，先停止
            if self.video_thread.running:
                self.video_thread.stop()
                self.video_thread.wait()  # 等待线程完全停止

            self.video_path = video_path
            # 移除路径标签更新
            # self.video_path_label.setText(f"视频文件：{video_path}")
            self.video_thread.set_video_path(video_path)
            self.video_thread.start()

    def on_video_processed(self, video_path):
        """视频处理完成回调"""
        # 更新状态为已处理
        if video_path in self.video_status:
            self.video_status[video_path]['status'] = "已处理"

        # 通知AutoProcessor当前视频已完成
        if hasattr(self, 'auto_processor') and self.auto_processor.running:
            self.auto_processor.video_finished(video_path)

        # 移除表格更新
        # self.update_video_table()

        # 移除计数显示重置
        # self.count_label.setText("0")

    # update_video_table 方法已删除

    def update_saved_images(self, image_path):
        """更新保存的图片路径列表"""
        self.saved_images.append(image_path)
        # 更新最后处理帧的时间
        if self.auto_processor.running:
            self.auto_processor.update_last_frame_time()

    def select_model(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.pt)"
        )
        if file_name:
            self.model_path = file_name
            # 移除标签更新
            # self.model_path_label.setText(f"模型文件：{self.model_path}")

    def select_output_dir(self):
        dir_name = QFileDialog.getExistingDirectory(
            self, "选择输出目录"
        )
        if dir_name:
            self.save_path = dir_name
            # 移除标签更新
            # self.output_path_label.setText(f"输出目录：{self.save_path}")

    def open_output_dir(self):
        if os.path.exists(self.save_path):
            if platform.system() == "Windows":
                os.startfile(self.save_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", self.save_path])
            else:  # Linux
                subprocess.run(["xdg-open", self.save_path])

    def select_video(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi)"
        )
        if file_name:
            self.video_path = file_name
            # 移除标签更新
            # self.video_path_label.setText(f"视频文件：{self.video_path}")

    def start_processing(self):
        """开始处理视频"""
        if not hasattr(self, 'video_path') or not self.video_path:
            # 如果没有选择视频文件，从自动处理文件夹中选择
            if not self.auto_folder_path:
                QMessageBox.warning(self, "警告", "请先选择自动处理文件夹")
                return
            
            # 从自动处理文件夹中找第一个视频文件
            import glob
            video_pattern = os.path.join(self.auto_folder_path, "**", "*.mp4")
            video_files = glob.glob(video_pattern, recursive=True)
            if not video_files:
                QMessageBox.warning(self, "警告", "在自动处理文件夹中未找到视频文件")
                return
            self.video_path = video_files[0]
            
        if self.video_thread.running:
            return
            
        # 设置处理参数
        self.video_thread.set_model_path(self.model_path)
        self.video_thread.set_save_path(self.save_path)
        self.video_thread.set_video_path(self.video_path)
        # 清空保存的图片列表
        self.saved_images = []
        self.video_thread.start()
        
    def stop_processing(self):
        """停止处理视频"""
        self.video_thread.stop()
        self.count_label.setText("0")

    def view_results(self):
        """打开结果查看器"""
        if not hasattr(self, 'result_viewer') or self.result_viewer is None:
            self.result_viewer = ResultViewerDialog(self)

        if hasattr(self, 'video_path') and self.video_path:
            # 获取父文件夹名称
            parent_folder = os.path.basename(os.path.dirname(self.video_path))
            result_dir = os.path.join(self.save_path, parent_folder)

            if os.path.exists(result_dir):
                self.result_viewer.load_images_from_directory(result_dir)
                self.result_viewer.show()
            else:
                # 如果目录不存在，显示一个提示框
                QMessageBox.information(self, "提示", f"找不到结果目录: {result_dir}")
        else:
            # 显示所有结果文件夹选择对话框
            dialog = VideoFolderSelector(self, self.save_path, "")
            if dialog.exec():
                selected_folder = dialog.get_selected_folder()
                if selected_folder:
                    folder_path = os.path.join(self.save_path, selected_folder)
                    if os.path.exists(folder_path):
                        self.result_viewer.load_images_from_directory(folder_path)
                        self.result_viewer.show()
                    else:
                        QMessageBox.information(self, "提示", f"找不到结果目录: {folder_path}")

    def browse_video_results(self):
        """浏览不同视频的结果文件夹"""
        # 获取当前视频的父文件夹名称（如果有）
        current_folder_name = ""
        if self.video_path:
            current_folder_name = os.path.basename(os.path.dirname(self.video_path))

        # 显示文件夹选择对话框
        dialog = VideoFolderSelector(self, self.save_path, current_folder_name)
        if dialog.exec():
            selected_folder = dialog.get_selected_folder()
            if selected_folder:
                folder_path = os.path.join(self.save_path, selected_folder)
                if os.path.exists(folder_path):
                    # 创建结果查看器如果不存在
                    if not self.result_viewer:
                        self.result_viewer = ResultViewerDialog(self)
                    # 加载选定文件夹的图片
                    self.result_viewer.load_images_from_directory(folder_path)
                    self.result_viewer.show()
                else:
                    QMessageBox.information(self, "提示", f"找不到结果目录: {folder_path}")

    # show_fullscreen 方法已删除

    # 恢复视频处理方法
    def update_image(self, cv_img):
        """更新图像显示"""
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def update_count(self, count):
        """更新计数显示"""
        self.count_label.setText(str(count))

    def update_fps(self, fps):
        """更新帧率显示"""
        self.fps_label.setText(f"{fps:.1f} FPS")

    def update_cemian_image(self, cv_img):
        """更新Cemian系统图像显示"""
        qt_img = self.convert_cv_qt_cemian(cv_img)
        self.cemian_video_label.setPixmap(qt_img)

    def update_cemian_count(self, count):
        """更新Cemian系统总计数显示"""
        # 这里可以添加总计数显示逻辑，如果需要的话
        pass

    def update_cemian_count_right_to_left(self, count):
        """更新Cemian系统从右往左计数显示"""
        # 这里可以添加从右往左计数显示逻辑，如果需要的话
        pass

    def update_cemian_count_left_to_right(self, count):
        """更新Cemian系统从左往右计数显示"""
        # 这里可以添加从左往右计数显示逻辑，如果需要的话
        pass

    def update_cemian_count_line1(self, count):
        """更新Cemian系统第一条线计数显示"""
        # 这里可以添加第一条线计数显示逻辑，如果需要的话
        pass

    def update_cemian_count_line2(self, count):
        """更新Cemian系统第二条线计数显示"""
        # 这里可以添加第二条线计数显示逻辑，如果需要的话
        pass

    def update_cemian_fps(self, fps):
        """更新Cemian系统帧率显示"""
        # 这里可以添加帧率显示逻辑，如果需要的话
        pass

    def update_cemian_saved_images(self, image_path):
        """更新Cemian系统保存的图片路径列表"""
        # 这里可以添加保存图片路径的逻辑，如果需要的话
        pass

    def convert_cv_qt(self, cv_img):
        """将OpenCV图像转换为QT图像"""
        import cv2
        from PySide6.QtGui import QImage, QPixmap
        
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_img = convert_to_Qt_format.scaled(800, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return QPixmap.fromImage(scaled_img)
        
    def convert_cv_qt_cemian(self, cv_img):
        """将OpenCV图像转换为QT图像（用于Cemian系统）"""
        import cv2
        from PySide6.QtGui import QImage, QPixmap
        
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # 根据Cemian视频标签的大小进行缩放
        scaled_img = convert_to_Qt_format.scaled(
            self.cemian_video_label.width(), 
            self.cemian_video_label.height(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        return QPixmap.fromImage(scaled_img)
        
    def start_processing(self):
        """开始处理视频"""
        if not hasattr(self, 'video_path') or not self.video_path or self.video_thread.running:
            QMessageBox.warning(self, "警告", "请先选择视频文件")
            return
            
        # 设置处理参数
        self.video_thread.set_model_path(self.model_path)
        self.video_thread.set_save_path(self.save_path)
        self.video_thread.set_video_path(self.video_path)
        # 清空保存的图片列表
        self.saved_images = []
        self.video_thread.start()
        
    def stop_processing(self):
        """停止处理视频"""
        self.video_thread.stop()
        self.count_label.setText("0")

    def open_train_info(self):
        """打开列车信息查看窗口"""
        # 如果窗口已存在且可见，则激活它；否则创建新窗口
        if hasattr(self, 'train_info_window') and self.train_info_window and self.train_info_window.isVisible():
            self.train_info_window.raise_()
            self.train_info_window.activateWindow()
        else:
            # 创建列车信息查看窗口
            self.train_info_window = TrainImageViewer(parent=self, output_dir=self.save_path)
            self.train_info_window.show()

    # convert_cv_qt 方法已删除

    def open_accuracy_visualization(self):
        """打开准确率可视化窗口"""
        try:
            if TrainAccuracyMainWindow is None:
                QMessageBox.critical(self, "错误", "准确率可视化模块未正确加载，请检查 accuracy_visualization.py 文件是否存在")
                return
            
            # 创建并显示准确率可视化窗口
            self.accuracy_window = TrainAccuracyMainWindow()
            self.accuracy_window.show()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开准确率可视化失败：{str(e)}")
    
    def open_water_detection(self):
        """打开积水识别系统"""
        try:
            # 创建积水识别对话框
            water_dialog = WaterDetectionDialog(self)
            if water_dialog.exec() == QDialog.Accepted:
                selected_mode = water_dialog.get_selected_mode()
                
                if selected_mode == "monitor":
                    self.start_water_monitoring()
                elif selected_mode == "batch":
                    self.start_water_batch_processing()
                    
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开积水识别系统失败：{str(e)}")
    
    def open_water_carriage_viewer(self):
        """打开积水车厢查看窗口"""
        try:
            # 如果窗口已存在且可见，则激活它；否则创建新窗口
            if hasattr(self, 'water_carriage_window') and self.water_carriage_window and self.water_carriage_window.isVisible():
                self.water_carriage_window.raise_()
                self.water_carriage_window.activateWindow()
            else:
                # 创建积水车厢查看窗口，传入self作为parent以便访问train_info_window
                self.water_carriage_window = WaterCarriageViewer(parent=self)
                self.water_carriage_window.show()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开积水车厢查看失败：{str(e)}")
    
    def start_water_monitoring(self):
        """启动积水识别监控模式"""
        try:
            # 已在运行则直接返回
            if self.water_monitor_running:
                QMessageBox.information(self, "信息", "积水识别监控已在运行中。")
                return True

            # 延迟导入，避免启动时阻塞
            from inference2 import WaterDetectionMonitor
            # 检查模型文件
            if not os.path.exists(self.water_model_path):
                QMessageBox.warning(self, "警告", f"积水识别模型文件不存在: {self.water_model_path}")
                return
            
            # 检查根文件夹
            if not os.path.exists(self.water_root_folder):
                QMessageBox.warning(self, "警告", f"根文件夹不存在: {self.water_root_folder}")
                return
            
            # 创建并启动监控器
            self.water_monitor = WaterDetectionMonitor(
                self.water_model_path, 
                self.water_root_folder, 
                self.water_processed_files, 
                self.water_check_interval
            )
            
            # 在后台启动监控
            threading.Thread(target=self.water_monitor.start_monitoring, daemon=True).start()
            self.water_monitor_running = True
            
            QMessageBox.information(self, "信息", "积水识别监控已启动！\n系统将自动检测新文件并进行积水识别。")
            return True
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动积水识别监控失败：{str(e)}")
            self.water_monitor_running = False
            return False
    
    def start_water_batch_processing(self):
        """启动积水识别批量处理模式"""
        try:
            # 延迟导入，避免启动时阻塞
            from inference2 import process_all_subfolders
            # 检查模型文件
            if not os.path.exists(self.water_model_path):
                QMessageBox.warning(self, "警告", f"积水识别模型文件不存在: {self.water_model_path}")
                return
            
            # 检查根文件夹
            if not os.path.exists(self.water_root_folder):
                QMessageBox.warning(self, "警告", f"根文件夹不存在: {self.water_root_folder}")
                return
            
            # 显示处理信息
            reply = QMessageBox.question(
                self, 
                "确认批量处理", 
                f"将对以下目录进行批量积水识别：\n{self.water_root_folder}\n\n此操作可能需要较长时间，是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # 在后台线程中执行批量处理
                def batch_process():
                    try:
                        process_all_subfolders(self.water_model_path, self.water_root_folder)
                        print("批量积水识别处理完成！")
                    except Exception as e:
                        print(f"批量处理失败：{str(e)}")
                
                # 创建后台线程
                process_thread = threading.Thread(target=batch_process, daemon=True)
                process_thread.start()
                
                QMessageBox.information(self, "信息", "批量积水识别已在后台启动！\n处理完成后请查看输出目录。")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动批量积水识别失败：{str(e)}")

    def open_download_manager(self):
        """直接调用data_download/main.py"""
        try:
            import subprocess
            import sys
            import os
            
            # 获取data_download/main.py的路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            download_main_path = os.path.join(current_dir, "data_download", "main.py")
            
            if os.path.exists(download_main_path):
                # 直接调用data_download/main.py
                subprocess.Popen(
                    [sys.executable, download_main_path], 
                    cwd=os.path.join(current_dir, "data_download")
                )
                QMessageBox.information(self, "信息", "视频下载管理器已启动！")
            else:
                QMessageBox.warning(self, "警告", f"找不到下载管理器文件：{download_main_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动下载管理器失败：{str(e)}")
    
    # show_download_status, open_download_folder, start_download_monitoring, stop_download_monitoring 方法已删除
    
    def closeEvent(self, event):
        """窗口关闭事件，清理资源"""
        reply = QMessageBox.question(
            self,
            "确认关闭",
            "确定要退出吗？\n将停止所有处理并关闭程序。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            event.ignore()
            return

        try:
            # 停止视频处理线程
            if hasattr(self, 'video_thread') and self.video_thread.running:
                self.video_thread.stop()
            
            # 停止自动处理器
            if hasattr(self, 'auto_processor') and self.auto_processor.running:
                self.auto_processor.stop()
            
            # 停止积水识别监控
            if hasattr(self, 'water_monitor_running') and self.water_monitor_running:
                try:
                    if hasattr(self, 'water_monitor') and self.water_monitor and hasattr(self.water_monitor, 'stop_monitoring'):
                        self.water_monitor.stop_monitoring()
                    self.water_monitor_running = False
                except Exception as wm_err:
                    print(f"停止积水识别监控时出错: {wm_err}")
            
            # 停止数据库自动更新器
            if hasattr(self, 'database_updater') and self.database_updater.running:
                self.database_updater.stop_monitoring()
            
            print("所有资源已清理")
            
        except Exception as e:
            print(f"清理资源时出错: {e}")
        
        event.accept()
    
    def cemian_start_auto_processing(self):
        """Cemian系统开始自动处理"""
        try:
            # 检查模型文件是否存在
            if not os.path.exists(self.cemian_model_path):
                QMessageBox.warning(self, "警告", f"Cemian检测模型文件不存在: {self.cemian_model_path}")
                return
            
            if not os.path.exists(self.cemian_plate_model_path):
                QMessageBox.warning(self, "警告", f"Cemian车牌识别模型文件不存在: {self.cemian_plate_model_path}")
                return
            
            # 设置模型路径
            self.cemian_video_thread.set_model_path(self.cemian_model_path)
            self.cemian_video_thread.set_plate_model_path(self.cemian_plate_model_path)
            
            # 设置保存路径
            self.cemian_video_thread.set_save_path(self.cemian_save_path)
            
            # 更新按钮状态和样式
            self.cemian_start_auto_btn.setEnabled(False)
            self.cemian_stop_auto_btn.setEnabled(True)
            
            # 改变按钮颜色以显示已激活状态
            self.cemian_start_auto_btn.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;  /* 绿色表示已激活 */
                    color: white;
                    border: none;
                    padding: 8px 12px;
                    border-radius: 4px;
                    font-weight: 500;
                    font-size: 12px;
                    min-width: 100px;
                    min-height: 24px;
                }
                QPushButton:disabled {
                    background-color: #28a745;  /* 保持绿色 */
                    color: white;
                }
            """)
            
            # 启动自动处理
            self.cemian_video_thread.start_auto_processing(self.cemian_video_folder)
            
            # 更新状态显示
            self.cemian_video_label.setText("🚀 侧面检测系统已启动\n\n正在处理视频...")
            
            print("侧面检测系统已启动")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动Cemian系统失败：{str(e)}")
    
    def cemian_stop_auto_processing(self):
        """Cemian系统停止自动处理"""
        try:
            # 停止Cemian视频处理线程
            if hasattr(self, 'cemian_video_thread') and self.cemian_video_thread:
                self.cemian_video_thread.stop_auto_processing()
            
            # 更新按钮状态和样式
            self.cemian_start_auto_btn.setEnabled(True)
            self.cemian_stop_auto_btn.setEnabled(False)
            
            # 恢复按钮原始样式
            button_style = """
                QPushButton {
                    background-color: #4a90e2;
                    color: white;
                    border: none;
                    padding: 8px 12px;
                    border-radius: 4px;
                    font-weight: 500;
                    font-size: 12px;
                    min-width: 100px;
                    min-height: 24px;
                }
                QPushButton:hover {
                    background-color: #357abd;
                }
                QPushButton:pressed {
                    background-color: #2c5f9e;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                    color: #666666;
                }
            """
            self.cemian_start_auto_btn.setStyleSheet(button_style)
            
            # 更新状态显示
            self.cemian_video_label.setText("📺 等待侧面检测视频...")
            
            print("侧面检测系统已停止")
            
        except Exception as e:
            print(f"停止Cemian系统时出错：{str(e)}")


# 列车数据缓存管理类
class TrainDataCache:
    def __init__(self):
        self.cache = {}  # 缓存字典: {cache_key: {data: ..., timestamp: ..., file_mtime: ...}}
        self.max_cache_size = 10  # 最多缓存10趟列车
        self.csv_cache = {}  # CSV文件缓存: {file_path: {data: ..., mtime: ..., indexed_by_image: {...}, indexed_by_plate: {...}}}
        
    def _get_cache_key(self, train_name, file_path):
        """生成缓存键，包含文件修改时间"""
        try:
            file_mtime = os.path.getmtime(file_path) if os.path.exists(file_path) else 0
            return f"{train_name}_{file_mtime}"
        except Exception:
            return f"{train_name}_0"
        
    def get_train_data(self, train_name, file_path, loader_func):
        """获取列车数据，优先从缓存读取，考虑文件修改时间"""
        cache_key = self._get_cache_key(train_name, file_path)
        
        # 检查缓存是否命中且有效
        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            try:
                current_file_mtime = os.path.getmtime(file_path) if os.path.exists(file_path) else 0
                if cached_item.get('file_mtime', 0) == current_file_mtime:
                    print(f"缓存命中: {train_name} (加载时间 < 100ms)")
                    return cached_item['data']
                else:
                    print(f"文件已更新，清除旧缓存: {train_name}")
                    del self.cache[cache_key]
            except Exception:
                # 如果获取文件时间出错，删除缓存
                del self.cache[cache_key]
        
        print(f"缓存未命中，开始加载: {train_name}")
        start_time = time.time()
        
        # 缓存未命中，加载数据
        data = loader_func(train_name, file_path)
        
        if data:
            # 管理缓存大小
            if len(self.cache) >= self.max_cache_size:
                # 删除最老的缓存项（简单FIFO策略）
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                print(f"缓存已满，删除最老缓存项: {oldest_key}")
            
            # 添加到缓存
            try:
                file_mtime = os.path.getmtime(file_path) if os.path.exists(file_path) else 0
            except Exception:
                file_mtime = 0
                
            self.cache[cache_key] = {
                'data': data,
                'timestamp': time.time(),
                'file_mtime': file_mtime
            }
            
            load_time = time.time() - start_time
            print(f"数据加载完成: {train_name} (耗时: {load_time:.2f}s)")
        
        return data
    
    def get_csv_data(self, csv_path, reload=False):
        """获取CSV数据，带缓存和文件修改时间检查"""
        if not os.path.exists(csv_path):
            return []
            
        try:
            current_mtime = os.path.getmtime(csv_path)
        except Exception:
            current_mtime = 0
            
        # 检查缓存
        if not reload and csv_path in self.csv_cache:
            cached_item = self.csv_cache[csv_path]
            if cached_item.get('mtime', 0) == current_mtime:
                return cached_item['data']
            else:
                print(f"CSV文件已更新，重新加载: {os.path.basename(csv_path)}")
        
        # 重新加载CSV数据
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as csv_file:
                reader = csv.DictReader(csv_file)
                data = list(reader)
                
            # 更新缓存
            self.csv_cache[csv_path] = {
                'data': data,
                'mtime': current_mtime,
                'timestamp': time.time()
            }
            
            # 只在调试模式打印
            # print(f"CSV文件加载完成: {os.path.basename(csv_path)} ({len(data)} 条记录)")
            return data
            
        except Exception as e:
            print(f"读取CSV文件失败: {csv_path}, 错误: {str(e)}")
            return []
    
    def get_csv_indexed_by_key(self, csv_path, key_field, reload=False):
        """获取CSV数据并建立字典索引，返回 (数据列表, 索引字典)"""
        data = self.get_csv_data(csv_path, reload)
        if not data:
            return [], {}
        
        # 建立索引字典，以指定字段为key
        index_dict = {}
        for row in data:
            key_value = row.get(key_field, '').strip()
            if key_value:
                # 支持多值索引（如果有重复key）
                if key_value not in index_dict:
                    index_dict[key_value] = row
                else:
                    # 如果已存在，保留第一个匹配项（可根据需要调整策略）
                    pass
        
        return data, index_dict
    
    def invalidate_cache(self):
        """清空所有缓存"""
        cache_count = len(self.cache)
        csv_cache_count = len(self.csv_cache)
        
        self.cache.clear()
        self.csv_cache.clear()
        
        print(f"缓存已清空: 列车数据缓存 {cache_count} 项, CSV缓存 {csv_cache_count} 项")
    
    def invalidate_train_cache(self, train_name):
        """清空特定列车的缓存"""
        # 找到所有与该列车相关的缓存键
        keys_to_remove = [key for key in self.cache.keys() if key.startswith(f"{train_name}_")]
        
        for key in keys_to_remove:
            del self.cache[key]
            
        if keys_to_remove:
            print(f"已清空列车 {train_name} 的缓存 ({len(keys_to_remove)} 项)")
    
    def get_cache_info(self):
        """获取缓存信息（用于调试）"""
        return {
            'train_cache_count': len(self.cache),
            'csv_cache_count': len(self.csv_cache),
            'max_cache_size': self.max_cache_size
        }


# 修改TrainImageViewer类
class TrainImageViewer(QMainWindow):
    def __init__(self, parent=None, output_dir=RUNS_OUTPUT_IMAGES_DIR):
        super().__init__(parent)
        self.setWindowTitle("列车信息与图片查看")
        # 设置窗口最小和初始大小，允许根据内容自适应
        self.setMinimumSize(1200, 600)
        self.setGeometry(100, 100, 1400, 700)
        self.output_dir = output_dir
        
        # 初始化缓存管理器
        self.cache_manager = TrainDataCache()
        
        # 确保data目录和rate.csv文件存在
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.rate_csv_path = os.path.join(self.data_dir, "rate.csv")
        self.init_rate_csv()

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

        self.init_ui()

        # 居中显示
        self.center_window()
        
    def start_auto_processing(self):
        """开始自动处理"""
        if not self.auto_folder_path:
            QMessageBox.warning(self, "警告", "请先选择自动处理文件夹")
            return

        # 设置模型和保存路径
        self.video_thread.set_model_path(self.model_path)
        self.video_thread.set_save_path(self.save_path)
        # 移除相关设置
        # self.video_thread.set_conf_threshold(self.conf_spinbox.value())
        # self.video_thread.set_detection_area(self.detection_area_checkbox.isChecked())

        # 启动自动处理器
        self.auto_processor.start()
        
        # 改变按钮颜色以显示已激活状态
        self.auto_start_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;  /* 绿色表示已激活 */
                color: white;
                border: none;
                padding: 14px 20px;
                border-radius: 6px;
                font-weight: 500;
                font-size: 13px;
                min-width: 140px;
            }
            QPushButton:disabled {
                background-color: #28a745;  /* 保持绿色 */
                color: white;
            }
        """)

    def stop_auto_processing(self):
        """停止自动处理"""
        self.auto_processor.stop()
        self.video_thread.stop()
        
        # 恢复按钮原始样式
        button_style = """
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 14px 20px;
                border-radius: 6px;
                font-weight: 500;
                font-size: 13px;
                min-width: 140px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2c5f9e;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """
        self.auto_start_btn.setStyleSheet(button_style)

    def open_water_detection(self):
        """打开积水识别系统"""
        try:
            # 已在运行则直接提示并保持绿色
            if self.water_monitor_running:
                self.water_detection_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #28a745;  /* 绿色表示已激活 */
                        color: white;
                        border: none;
                        padding: 12px 16px;
                        border-radius: 6px;
                        font-weight: 500;
                        font-size: 12px;
                        min-width: 110px;
                    }
                    QPushButton:disabled {
                        background-color: #28a745;
                        color: white;
                    }
                """)
                QMessageBox.information(self, "信息", "积水识别监控已在运行中。")
                return

            # 直接启动监控模式
            if self.start_water_monitoring():
                # 成功后按钮置绿，表示监控中
                self.water_detection_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #28a745;  /* 绿色表示已激活 */
                        color: white;
                        border: none;
                        padding: 12px 16px;
                        border-radius: 6px;
                        font-weight: 500;
                        font-size: 12px;
                        min-width: 110px;
                    }
                    QPushButton:disabled {
                        background-color: #28a745;
                        color: white;
                    }
                """)
            else:
                # 启动失败时恢复原样式
                view_button_style = """
                    QPushButton {
                        background-color: #4a90e2;
                        color: white;
                        border: none;
                        padding: 12px 16px;
                        border-radius: 6px;
                        font-weight: 500;
                        font-size: 12px;
                        min-width: 110px;
                    }
                    QPushButton:hover {
                        background-color: #357abd;
                    }
                    QPushButton:pressed {
                        background-color: #2c5f9e;
                    }
                    QPushButton:disabled {
                        background-color: #cccccc;
                        color: #666666;
                    }
                """
                self.water_detection_btn.setStyleSheet(view_button_style)

        except Exception as e:
            # 恢复按钮原始样式
            view_button_style = """
                QPushButton {
                    background-color: #4a90e2;
                    color: white;
                    border: none;
                    padding: 12px 16px;
                    border-radius: 6px;
                    font-weight: 500;
                    font-size: 12px;
                    min-width: 110px;
                }
                QPushButton:hover {
                    background-color: #357abd;
                }
                QPushButton:pressed {
                    background-color: #2c5f9e;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                    color: #666666;
                }
            """
            self.water_detection_btn.setStyleSheet(view_button_style)
            
            QMessageBox.critical(self, "错误", f"打开积水识别系统失败：{str(e)}")
    
    def start_water_monitoring(self):
        """启动积水识别监控模式"""
        try:
            # 检查模型文件
            if not os.path.exists(self.water_model_path):
                QMessageBox.warning(self, "警告", f"积水识别模型文件不存在: {self.water_model_path}")
                return
            
            # 检查根文件夹
            if not os.path.exists(self.water_root_folder):
                QMessageBox.warning(self, "警告", f"根文件夹不存在: {self.water_root_folder}")
                return
            
            # 创建并启动监控器
            self.water_monitor = WaterDetectionMonitor(
                self.water_model_path, 
                self.water_root_folder, 
                self.water_processed_files, 
                self.water_check_interval
            )
            
            # 在后台启动监控
            threading.Thread(target=self.water_monitor.start_monitoring, daemon=True).start()
            
            QMessageBox.information(self, "信息", "积水识别监控已启动！\n系统将自动检测新文件并进行积水识别。")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动积水识别监控失败：{str(e)}")
    
    def start_water_batch_processing(self):
        """启动积水识别批量处理模式"""
        try:
            # 检查模型文件
            if not os.path.exists(self.water_model_path):
                QMessageBox.warning(self, "警告", f"积水识别模型文件不存在: {self.water_model_path}")
                return
            
            # 检查根文件夹
            if not os.path.exists(self.water_root_folder):
                QMessageBox.warning(self, "警告", f"根文件夹不存在: {self.water_root_folder}")
                return
            
            # 显示处理信息
            reply = QMessageBox.question(
                self, 
                "确认批量处理", 
                f"将对以下目录进行批量积水识别：\n{self.water_root_folder}\n\n此操作可能需要较长时间，是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # 在后台线程中执行批量处理
                def batch_process():
                    try:
                        process_all_subfolders(self.water_model_path, self.water_root_folder)
                        print("批量积水识别处理完成！")
                    except Exception as e:
                        print(f"批量处理失败：{str(e)}")
                
                # 创建后台线程
                process_thread = threading.Thread(target=batch_process, daemon=True)
                process_thread.start()
                
                QMessageBox.information(self, "信息", "批量积水识别已在后台启动！\n处理完成后请查看输出目录。")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动批量积水识别失败：{str(e)}")

        # 确保data目录和rate.csv文件存在
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.rate_csv_path = os.path.join(self.data_dir, "rate.csv")
        self.init_rate_csv()

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

        self.init_ui()

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
    
    def adjust_window_height(self):
        """根据表格内容动态调整窗口高度"""
        try:
            # 获取表格行数
            row_count = self.carriage_table.rowCount()
            
            # 如果表格为空，使用最小高度
            if row_count == 0:
                min_height = 600
            else:
                # 计算表格所需高度
                # 表头高度 + 行高 * 行数 + 边距和控件高度
                header_height = self.carriage_table.horizontalHeader().height()
                row_height = self.carriage_table.verticalHeader().defaultSectionSize()
                table_height = header_height + (row_height * row_count)
                
                # 计算其他控件的高度（顶部控件、搜索框等）
                other_controls_height = 200  # 包括选择区域、搜索框、准确率显示等
                
                # 计算总高度，加上边距
                content_height = table_height + other_controls_height + 80  # 80是边距和间距
                
                # 限制最大高度（不超过屏幕高度的90%）
                screen = QApplication.primaryScreen()
                screen_height = screen.geometry().height()
                max_height = int(screen_height * 0.9)
                
                min_height = min(content_height, max_height)
                min_height = max(min_height, 600)  # 最小高度600
            
            # 获取当前窗口宽度，保持宽度不变
            current_width = self.width()
            
            # 调整窗口大小
            self.resize(current_width, min_height)
            
            # 重新居中显示
            self.center_window()
            
        except Exception as e:
            print(f"调整窗口高度时出错: {str(e)}")

    def init_ui(self):
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QHBoxLayout(central_widget)  # 改为水平布局
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)  # 减少边距，减少留白

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
        left_layout.setContentsMargins(12, 12, 12, 12)  # 减少内部边距
        left_layout.setSpacing(8)  # 减少控件间距

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

        # 修改刷新方法，清空缓存
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.refresh_with_cache_clear)
        train_select_layout.addWidget(self.refresh_btn)

        left_layout.addLayout(train_select_layout)

        # 搜索框
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("搜索:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入关键字搜索...")
        self.search_input.textChanged.connect(self.filter_table)
        self.search_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
            }
        """)
        search_layout.addWidget(self.search_input)

        left_layout.addLayout(search_layout)

        # 车厢信息表格
        self.carriage_table = QTableWidget()
        self.carriage_table.setColumnCount(6)  # 增加积水面积占比列
        self.carriage_table.setHorizontalHeaderLabels(["序号", "车号", "识别车号", "识别时间", "积水信息", "积水面积占比(%)"])
        self.carriage_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)  # 序号列自适应
        self.carriage_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.carriage_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        # 为识别时间列设置固定宽度，确保能完整显示时间信息
        self.carriage_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.carriage_table.setColumnWidth(3, 220)  # 设置识别时间列宽度为220像素，确保能完整显示"2024-01-01 12:34:56"格式的时间
        self.carriage_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.carriage_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)  # 积水面积占比列自适应
        self.carriage_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.carriage_table.setSelectionMode(QTableWidget.SingleSelection)
        self.carriage_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.carriage_table.setAlternatingRowColors(True)
        self.carriage_table.currentCellChanged.connect(self.on_table_row_selected)
        
        # 设置行高，根据内容自适应
        self.carriage_table.verticalHeader().setDefaultSectionSize(35)  # 设置行高为35像素
        self.carriage_table.verticalHeader().setVisible(False)  # 隐藏行号，减少留白

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
        
        # 设置表头文本居中对齐
        for col in range(self.carriage_table.columnCount()):
            header_item = self.carriage_table.horizontalHeaderItem(col)
            if header_item:
                header_item.setTextAlignment(Qt.AlignCenter)

        left_layout.addWidget(self.carriage_table, 1)  # 设置拉伸因子为1，使表格占据更多空间

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
        right_layout.setContentsMargins(12, 12, 12, 12)  # 减少边距
        right_layout.setSpacing(8)  # 减少间距

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
        self.accuracy_circle.setAlignment(Qt.AlignCenter)
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
        self.image_scroll_area.setFrameShape(QScrollArea.NoFrame)
        self.image_scroll_area.setMinimumHeight(300)  # 减少最小高度，减少留白
        self.image_scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
        """)

        self.image_label = QLabel("无图片")
        self.image_label.setAlignment(Qt.AlignCenter)
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
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: black;")
        images_layout.addWidget(separator)

        # 侧面图片区域
        self.side_image_scroll_area = QScrollArea()
        self.side_image_scroll_area.setWidgetResizable(True)
        self.side_image_scroll_area.setFrameShape(QScrollArea.NoFrame)
        self.side_image_scroll_area.setMinimumHeight(300)  # 减少最小高度，减少留白
        self.side_image_scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
        """)

        self.side_image_label = QLabel("无侧面图片")
        self.side_image_label.setAlignment(Qt.AlignCenter)
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
        main_layout.addWidget(left_panel, 2)  # 左侧占2
        main_layout.addWidget(right_panel, 3)  # 右侧占3

        # 加载列车文件
        self.load_train_files()

    def init_rate_csv(self):
        """初始化rate.csv文件，如果不存在则创建表头"""
        try:
            if not os.path.exists(self.rate_csv_path):
                with open(self.rate_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['列车名称', '总车厢数', '识别正确数', '参与统计数', '准确率(%)', '统计时间']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                print(f"创建新的rate.csv文件: {self.rate_csv_path}")
        except Exception as e:
            print(f"初始化rate.csv文件失败: {str(e)}")
    
    def save_train_statistics_to_csv(self, train_name, total_carriages, correct_count, normal_carriages_count, accuracy_rate):
        """保存列车统计信息到CSV文件，自动更新或新增记录"""
        try:
            # 检查CSV文件是否存在
            file_exists = os.path.exists(self.rate_csv_path)
            
            # 如果文件存在，读取现有数据
            existing_data = []
            if file_exists:
                try:
                    with open(self.rate_csv_path, 'r', encoding='utf-8') as csvfile:
                        reader = csv.DictReader(csvfile)
                        existing_data = list(reader)
                except Exception as e:
                    print(f"读取现有CSV数据失败: {str(e)}")
                    existing_data = []
            
            # 准备新的统计记录
            new_record = {
                '列车名称': train_name,
                '总车厢数': total_carriages,
                '识别正确数': correct_count,
                '参与统计数': normal_carriages_count,
                '准确率(%)': f"{accuracy_rate:.2f}",
                '统计时间': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 检查是否已存在该列车的记录
            train_exists = False
            for i, record in enumerate(existing_data):
                if record.get('列车名称', '').strip() == train_name.strip():
                    # 更新现有记录
                    existing_data[i] = new_record
                    train_exists = True
                    print(f"更新列车 {train_name} 的统计信息")
                    break
            
            # 如果不存在，添加新记录
            if not train_exists:
                existing_data.append(new_record)
                print(f"新增列车 {train_name} 的统计信息")
            
            # 重新写入整个文件
            with open(self.rate_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['列车名称', '总车厢数', '识别正确数', '参与统计数', '准确率(%)', '统计时间']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_data)
                
            print(f"统计信息已保存到: {self.rate_csv_path}")
            
        except Exception as e:
            print(f"保存统计信息失败: {str(e)}")

    def load_train_data_with_cache(self, train_name, file_path):
        """带缓存的列车数据加载方法（优化版：批量加载CSV并建立索引）"""
        # 读取列车信息
        if not os.path.exists(file_path):
            return None
        
        # 预先加载并建立CSV索引，避免在循环中重复查找
        # 1. 加载积水识别结果CSV索引（以图片名为key）
        individual_csv = os.path.join(RUNS_OUTPUT_IMAGES_DIR, train_name, f"{train_name}_积水识别结果.csv")
        water_summary_csv = os.path.join(RUNS_OUTPUT_IMAGES_DIR, "所有子文件夹积水识别结果汇总.csv")
        
        water_csv_index = {}  # {image_name: row}
        water_csv_path = None
        if os.path.exists(individual_csv):
            csv_data = self.cache_manager.get_csv_data(individual_csv)
            water_csv_path = individual_csv
            # 建立索引
            for row in csv_data:
                image_name = row.get('原图片名称', '').strip()
                if image_name:
                    water_csv_index[image_name] = row
        elif os.path.exists(water_summary_csv):
            csv_data = self.cache_manager.get_csv_data(water_summary_csv)
            water_csv_path = water_summary_csv
            # 建立索引（汇总文件需要考虑子文件夹）
            for row in csv_data:
                subfolder = row.get('子文件夹', '').strip()
                image_name = row.get('原图片名称', '').strip()
                if image_name and subfolder == train_name:
                    water_csv_index[image_name] = row
        
        # 2. 加载侧面图片CSV索引（以车号为key）
        side_csv_path = os.path.join(OUTPUT_IMAGES_DIR, train_name, "plate_results.csv")
        side_csv_data = self.cache_manager.get_csv_data(side_csv_path) if os.path.exists(side_csv_path) else []
        side_csv_index = {}  # {plate_number_upper: row}
        side_csv_digits_index = {}  # {extracted_digits: [row, ...]} 用于相似匹配
        
        for row in side_csv_data:
            plate_number = row.get('plate_number', '').strip()
            if plate_number:
                # 完全匹配索引
                side_csv_index[plate_number.upper()] = row
                # 数字序列索引（用于相似匹配）
                digits = self.extract_digits(plate_number)
                if digits:
                    if digits not in side_csv_digits_index:
                        side_csv_digits_index[digits] = []
                    side_csv_digits_index[digits].append(row)
        
        # 分别存储普通车号和特殊车号
        normal_carriages = []
        special_carriages = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
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
                        
                        # 使用索引快速查找顶部图片和积水信息
                        carriage_num = int(carriage_id) if carriage_id.isdigit() else 0
                        image_name = f"frame_{carriage_num}.jpg"
                        top_image_info = self.find_image_info_from_index(
                            train_name, image_name, water_csv_index, water_csv_path
                        )
                        
                        # 使用索引快速查找侧面图片和车号识别结果
                        side_image_info = self.find_side_image_from_index(
                            train_name, full_carriage_number, side_csv_index, side_csv_digits_index
                        )
                        
                        recognized_plate = "未识别"
                        recognized_time = "未知"
                        
                        if side_image_info:
                            match_type = side_image_info.get('match_type', 'unknown')
                            if match_type == 'exact':
                                recognized_plate = side_image_info['plate_number']
                            elif match_type == 'similar':
                                similarity_score = side_image_info.get('similarity_score', 0)
                                if similarity_score >= 4:
                                    recognized_plate = side_image_info['plate_number']
                                else:
                                    recognized_plate = full_carriage_number
                            elif match_type == 'rfid':
                                recognized_plate = side_image_info['plate_number']
                            else:
                                recognized_plate = side_image_info.get('plate_number', '未识别')
                            
                            recognized_time = side_image_info.get('timestamp', '未知')
                        
                        # 创建车厢信息字典
                        carriage_info = {
                            'carriage_id': carriage_id,
                            'full_carriage_number': full_carriage_number,
                            'recognized_plate': recognized_plate,
                            'recognized_time': recognized_time,
                            'top_image_info': top_image_info,
                            'side_image_info': side_image_info,
                            'is_special': is_special
                        }
                        
                        # 根据是否为特殊车号分类存储
                        if is_special:
                            special_carriages.append(carriage_info)
                        else:
                            normal_carriages.append(carriage_info)
            
            # 先添加普通车号，再添加特殊车号
            all_carriages = normal_carriages + special_carriages
            
            # 计算准确率
            accuracy_rate = self.calculate_accuracy_rate_cached(train_name, normal_carriages)
            
            return {
                'carriages': all_carriages,
                'accuracy_rate': accuracy_rate,
                'load_time': time.time()
            }
            
        except Exception as e:
            print(f"加载列车数据失败: {str(e)}")
            return None

    def find_image_info_from_index(self, train_name, image_name, water_csv_index, water_csv_path):
        """使用索引快速查找图片信息（优化版）"""
        try:
            # 构建完整图片路径
            image_path = os.path.join(self.output_dir, train_name, image_name)
            
            has_water = False
            water_detected_image_path = None
            water_area_ratio = 0.0
            
            # 使用索引直接查找（O(1)复杂度）
            row = water_csv_index.get(image_name)
            if row:
                has_water = row.get('是否有积水', '').strip() == '是'
                
                # 获取积水面积占比
                try:
                    raw_ratio = float(row.get('积水面积占比(%)', '0.0'))
                    water_area_ratio = raw_ratio * 1.3 if has_water else 0.0
                except (ValueError, TypeError):
                    water_area_ratio = 0.0
                
                if has_water and row.get('识别后图片路径', '').strip():
                    water_detected_image_path = row.get('识别后图片路径', '').strip()
                    if not os.path.isabs(water_detected_image_path):
                        water_detected_image_path = os.path.abspath(water_detected_image_path)
            
            # 减少文件存在性检查（延迟到实际使用时）
            return {
                'original_path': image_path,  # 延迟检查是否存在
                'water_detected_path': water_detected_image_path,
                'has_water': has_water,
                'image_name': image_name,
                'water_area_ratio': water_area_ratio
            }
        except Exception as e:
            return {
                'original_path': None,
                'water_detected_path': None,
                'has_water': False,
                'image_name': None,
                'water_area_ratio': 0.0
            }
    
    def find_side_image_from_index(self, train_name, full_carriage_number, side_csv_index, side_csv_digits_index):
        """使用索引快速查找侧面图片信息（优化版）"""
        try:
            # 第一步：完全匹配（O(1)复杂度）
            row = side_csv_index.get(full_carriage_number.upper())
            if row:
                return {
                    'image_path': os.path.join(OUTPUT_IMAGES_DIR, train_name, row['image_name']),
                    'plate_number': row['plate_number'],
                    'timestamp': row['timestamp'],
                    'match_type': 'exact'
                }
            
            # 第二步：相似车号匹配（优化：只在有数字序列时进行）
            target_digits = self.extract_digits(full_carriage_number)
            if len(target_digits) >= 5:
                best_match = None
                max_common_digits = 0
                
                # 遍历所有可能的数字序列匹配
                for digits, rows in side_csv_digits_index.items():
                    if len(digits) >= 4:  # 至少4位数字才考虑
                        common_digits = self.count_common_consecutive_digits(target_digits, digits)
                        if common_digits >= 4 and common_digits > max_common_digits:
                            # 选择第一个匹配的行
                            row = rows[0]
                            max_common_digits = common_digits
                            best_match = {
                                'image_path': os.path.join(OUTPUT_IMAGES_DIR, train_name, row['image_name']),
                                'plate_number': full_carriage_number,
                                'timestamp': row['timestamp'],
                                'match_type': 'similar',
                                'similarity_score': common_digits
                            }
                
                if best_match:
                    return best_match
            
            # 第三步：使用RFID兜底
            return self.get_rfid_fallback_data(train_name, full_carriage_number)
            
        except Exception as e:
            return self.get_rfid_fallback_data(train_name, full_carriage_number)

    def find_image_for_carriage_with_csv_cached(self, train_name, carriage_id):
        """缓存版本的图片查找方法（优先读取个别CSV文件）"""
        try:
            # 将车厢编号转换为整数（去除前导零）
            carriage_num = int(carriage_id)
            
            # 构建对应的frame_X.jpg文件名
            image_name = f"frame_{carriage_num}.jpg"
            
            # 构建完整图片路径
            image_path = os.path.join(self.output_dir, train_name, image_name)
            
            has_water = False
            water_detected_image_path = None
            water_area_ratio = 0.0  # 新增积水面积占比
            
            # 第一优先级：读取各列车文件夹下的积水识别结果CSV
            individual_csv = os.path.join(RUNS_OUTPUT_IMAGES_DIR, train_name, f"{train_name}_积水识别结果.csv")
            csv_data = []
            
            if os.path.exists(individual_csv):
                # print(f"优先读取个别积水识别结果文件: {individual_csv}")
                csv_data = self.cache_manager.get_csv_data(individual_csv)
            else:
                # 第二优先级：回退到汇总文件
                water_summary_csv = os.path.join(RUNS_OUTPUT_IMAGES_DIR, "所有子文件夹积水识别结果汇总.csv")
                # print(f"回退到汇总文件: {water_summary_csv}")
                csv_data = self.cache_manager.get_csv_data(water_summary_csv)
            
            # 解析CSV数据
            for row in csv_data:
                # 对于个别列车的CSV文件，不需要匹配子文件夹，只需匹配原图片名称
                original_name_key = '原图片名称'
                water_status_key = '是否有积水'
                detected_path_key = '识别后图片路径' 
                water_ratio_key = '积水面积占比(%)'
                
                # 检查是否匹配当前图片
                if row.get(original_name_key, '').strip() == image_name:
                    has_water = row.get(water_status_key, '').strip() == '是'
                    
                    # 获取积水面积占比
                    try:
                        raw_ratio = float(row.get(water_ratio_key, '0.0'))
                        # 根据规范：积水面积占比需要乘以1.3，无水时保持为0
                        water_area_ratio = raw_ratio * 1.3 if has_water else 0.0
                    except (ValueError, TypeError):
                        water_area_ratio = 0.0
                    
                    if has_water and row.get(detected_path_key, '').strip():
                        # 构建积水识别图片的完整路径
                        water_detected_image_path = row.get(detected_path_key, '').strip()
                        # 如果是相对路径，转换为绝对路径
                        if not os.path.isabs(water_detected_image_path):
                            water_detected_image_path = os.path.abspath(water_detected_image_path)
                    
                    # 减少print语句以提高性能
                    # print(f"从{individual_csv if os.path.exists(individual_csv) else '汇总文件'}中找到匹配记录: {image_name}, 有积水: {has_water}, 面积占比: {water_area_ratio:.2f}%")
                    break
            
            # 返回图片信息字典
            return {
                'original_path': image_path if os.path.exists(image_path) else None,
                'water_detected_path': water_detected_image_path,
                'has_water': has_water,
                'image_name': image_name,
                'water_area_ratio': water_area_ratio  # 新增积水面积占比
            }
        except ValueError:
            return {
                'original_path': None,
                'water_detected_path': None,
                'has_water': False,
                'image_name': None,
                'water_area_ratio': 0.0
            }
    
    def find_side_image_by_plate_number_cached(self, train_name, full_carriage_number):
        """缓存版本的侧面图片查找方法（支持相似车号匹配）"""
        try:
            # 构建CSV文件路径（仅查找同名文件夹）
            csv_path = os.path.join("output_images", train_name, "plate_results.csv")
            
            # 使用缓存读取CSV数据
            csv_data = self.cache_manager.get_csv_data(csv_path)
            
            if not csv_data:
                return self.get_rfid_fallback_data(train_name, full_carriage_number)
            
            # 第一步：完全匹配
            for row in csv_data:
                if row.get('plate_number', '').strip().upper() == full_carriage_number.strip().upper():
                    return {
                        'image_path': os.path.join("output_images", train_name, row['image_name']),
                        'plate_number': row['plate_number'],
                        'timestamp': row['timestamp'],
                        'match_type': 'exact'  # 完全匹配
                    }
            
            # 第二步：相似车号匹配（>=5位数字相同）
            target_digits = self.extract_digits(full_carriage_number)
            if len(target_digits) >= 5:
                best_match = None
                max_common_digits = 0
                
                for row in csv_data:
                    recognized_plate = row.get('plate_number', '').strip()
                    if recognized_plate:
                        recognized_digits = self.extract_digits(recognized_plate)
                        common_digits = self.count_common_consecutive_digits(target_digits, recognized_digits)
                        
                        # 至少有4个连续数字相同
                        if common_digits >= 4 and common_digits > max_common_digits:
                            max_common_digits = common_digits
                            best_match = {
                                'image_path': os.path.join("output_images", train_name, row['image_name']),
                                'plate_number': full_carriage_number,  # 显示原车号，不加RFID前缀
                                'timestamp': row['timestamp'],
                                'match_type': 'similar',  # 相似匹配
                                'similarity_score': common_digits
                            }
                
                if best_match:
                    return best_match
            
            # 第三步：使用RFID兜底（<5位相似或无匹配）
            return self.get_rfid_fallback_data(train_name, full_carriage_number)
            
        except Exception as e:
            print(f"查找侧面图片信息时出错: {str(e)}")
            return self.get_rfid_fallback_data(train_name, full_carriage_number)
    
    def extract_digits(self, text):
        """提取文本中的数字序列"""
        import re
        return ''.join(re.findall(r'\d', text))
    
    def count_common_consecutive_digits(self, digits1, digits2):
        """计算两个数字序列中最长的连续相同数字个数"""
        if not digits1 or not digits2:
            return 0
        
        max_length = 0
        for i in range(len(digits1)):
            for j in range(len(digits2)):
                length = 0
                while (i + length < len(digits1) and 
                       j + length < len(digits2) and 
                       digits1[i + length] == digits2[j + length]):
                    length += 1
                max_length = max(max_length, length)
        
        return max_length
    
    def get_rfid_fallback_data(self, train_name, full_carriage_number):
        """获取RFID兜底数据（如果找不到对应图片）"""
        return {
            'image_path': None,  # 没有对应图片
            'plate_number': full_carriage_number,  # 直接使用原车号，不加RFID前缀
            'timestamp': "未知",
            'match_type': 'rfid'  # RFID兜底
        }
    
    def calculate_accuracy_rate_cached(self, train_name, normal_carriages):
        """缓存版本的准确率计算方法（优化版）"""
        try:
            if not normal_carriages:
                return 0.0
            
            total_carriages = len(normal_carriages)
            correct_count = 0
            exact_match_count = 0
            high_similar_count = 0
            rfid_fallback_count = 0
            no_image_count = 0
            
            for carriage in normal_carriages:
                side_image_info = carriage.get('side_image_info')
                
                if not side_image_info:
                    # 没有侧面图片信息，算作错误
                    no_image_count += 1
                    continue
                
                match_type = side_image_info.get('match_type', 'unknown')
                
                # 根据匹配类型计算准确率
                if match_type == 'exact':
                    # 完全匹配，算作正确
                    correct_count += 1
                    exact_match_count += 1
                elif match_type == 'similar':
                    similarity_score = side_image_info.get('similarity_score', 0)
                    if similarity_score >= 4:
                        # >=4位相似，算作正确
                        correct_count += 1
                        high_similar_count += 1
                    else:
                        # <5位相似，归入RFID兜底，算作错误
                        rfid_fallback_count += 1
                elif match_type == 'rfid':
                    # RFID兜底，算作错误
                    rfid_fallback_count += 1
                else:
                    # 未知匹配类型，算作错误
                    no_image_count += 1
            
            # 计算最终准确率
            accuracy_rate = (correct_count / total_carriages) * 100
            
            return accuracy_rate
            
        except Exception as e:
            print(f"计算准确率时出错: {str(e)}")
            return 0.0

    def load_train_files(self):
        """加载列车文件并默认选择最新的一个，智能增量更新rate.csv"""
        # 清空下拉框
        self.train_combo.clear()

        # 加载列车文件
        train_dir = "F:\\baowen"  # 与main.py保持一致
        if not os.path.exists(train_dir):
            QMessageBox.warning(self, "警告", f"列车信息目录不存在: {train_dir}")
            return

        # 获取所有txt文件
        txt_files = [
            f for f in os.listdir(train_dir)
            if f.lower().endswith('.txt') and os.path.isfile(os.path.join(train_dir, f))
        ]

        if not txt_files:
            return

        # 获取文件修改时间，用于排序
        file_times = []
        for txt_file in txt_files:
            file_path = os.path.join(train_dir, txt_file)
            mod_time = os.path.getmtime(file_path)
            file_times.append((txt_file, mod_time))

        # 按修改时间倒序排序
        file_times.sort(key=lambda x: x[1], reverse=True)

        # 添加到下拉框
        for txt_file, _ in file_times:
            file_name = os.path.splitext(txt_file)[0]  # 去掉.txt后缀
            self.train_combo.addItem(file_name, os.path.join(train_dir, txt_file))

        # 默认选择最新的文件（第一个）
        if self.train_combo.count() > 0:
            self.train_combo.setCurrentIndex(0)
        
        # 更新按钮状态
        self.update_navigation_buttons()
        
        # 智能增量更新rate.csv（只处理新增列车）
        print("检查是否有新列车需要加入rate.csv...")
        self.smart_update_csv_in_background()

    def on_train_selected(self, index):
        """当选择列车时（使用缓存优化）"""
        if index < 0:
            return

        # 更新导航按钮状态
        self.update_navigation_buttons()

        # 获取文件路径和列车名称
        file_path = self.train_combo.currentData()
        train_name = self.train_combo.currentText()

        if not os.path.exists(file_path):
            return

        # 使用缓存加载数据
        train_data = self.cache_manager.get_train_data(
            train_name, 
            file_path, 
            self.load_train_data_with_cache
        )
        
        if not train_data:
            return
        
        # 保存当前列车数据供界面显示使用
        self.current_train_data = train_data
        
        # 更新界面显示
        self.update_ui_with_train_data(train_data)
        
        # 更新导航按钮状态
        self.update_navigation_buttons()
    
    def update_ui_with_train_data(self, train_data):
        """使用缓存数据更新界面（优化版：批量更新表格）"""
        # 检查并更新汇总文件（如果需要）
        self.auto_update_summary_if_needed()
        
        # 获取车厢数据
        all_carriages = train_data['carriages']
        accuracy_rate = train_data['accuracy_rate']
        
        # 禁用表格更新以提高性能
        self.carriage_table.setUpdatesEnabled(False)
        
        try:
            # 批量设置行数，而不是逐个插入
            self.carriage_table.setRowCount(len(all_carriages))
            
            # 批量填充表格数据
            for row_idx, carriage_info in enumerate(all_carriages):
                # 设置表格项，并设置文本居中对齐
                item0 = QTableWidgetItem(carriage_info['carriage_id'])
                item0.setTextAlignment(Qt.AlignCenter)
                self.carriage_table.setItem(row_idx, 0, item0)
                
                item1 = QTableWidgetItem(carriage_info['full_carriage_number'])
                item1.setTextAlignment(Qt.AlignCenter)
                self.carriage_table.setItem(row_idx, 1, item1)
                
                item2 = QTableWidgetItem(carriage_info['recognized_plate'])
                item2.setTextAlignment(Qt.AlignCenter)
                self.carriage_table.setItem(row_idx, 2, item2)
                
                item3 = QTableWidgetItem(carriage_info['recognized_time'])
                item3.setTextAlignment(Qt.AlignCenter)
                self.carriage_table.setItem(row_idx, 3, item3)
                
                # 设置积水信息
                water_status = "有水" if carriage_info['top_image_info']['has_water'] else "无水"
                item4 = QTableWidgetItem(water_status)
                item4.setTextAlignment(Qt.AlignCenter)
                self.carriage_table.setItem(row_idx, 4, item4)
                
                # 设置积水面积占比
                water_ratio = carriage_info['top_image_info'].get('water_area_ratio', 0.0)
                water_ratio_text = f"{water_ratio:.2f}%" if water_ratio > 0 else "0.0%"
                item5 = QTableWidgetItem(water_ratio_text)
                item5.setTextAlignment(Qt.AlignCenter)
                self.carriage_table.setItem(row_idx, 5, item5)

                # 存储图片路径数据
                item0.setData(Qt.UserRole, carriage_info['top_image_info'])

                if carriage_info['side_image_info']:
                    item2.setData(Qt.UserRole, carriage_info['side_image_info']['image_path'])
                else:
                    item2.setData(Qt.UserRole, None)

                # 为特殊车号设置不同的背景色
                if carriage_info['is_special']:
                    for col in range(self.carriage_table.columnCount()):
                        item = self.carriage_table.item(row_idx, col)
                        if item:
                            item.setBackground(QColor(255, 255, 255))
        finally:
            # 重新启用更新
            self.carriage_table.setUpdatesEnabled(True)

        # 如果有数据，选择第一行
        if self.carriage_table.rowCount() > 0:
            self.carriage_table.selectRow(0)
        
        # 根据表格内容动态调整窗口高度
        self.adjust_window_height()

        # 更新准确率显示
        self.update_accuracy_display(accuracy_rate)
    
    def update_accuracy_display(self, accuracy_rate, train_data=None):
        """更新准确率显示（简洁版）"""
        # 更新准确率显示
        self.accuracy_circle.setText(f"{accuracy_rate:.1f}%")

        # 只显示简单的准确率信息
        self.page_accuracy_info.setText(f"当前页准确率: {accuracy_rate:.1f}%")
        
        # 仅在需要时自动写入CSV文件（避免重复写入）
        self.save_accuracy_to_csv_if_needed(accuracy_rate)

        # 根据准确率设置不同的颜色 - 白色背景简洁设计
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
    
    def save_accuracy_to_csv_if_needed(self, accuracy_rate):
        """仅在需要时才写入CSV（避免重复写入）"""
        try:
            current_train = self.train_combo.currentText()
            if not current_train:
                return
                
            # 检查rate.csv中是否已存在该列车的记录
            if self.is_train_already_in_csv(current_train):
                # 如果已存在且数据一致，则不需要更新
                existing_accuracy = self.get_existing_accuracy_from_csv(current_train)
                if existing_accuracy is not None and abs(existing_accuracy - accuracy_rate) < 0.1:
                    print(f"列车 {current_train} 的rate.csv记录已存在且数据一致，跳过写入")
                    return
                    
            # 如果不存在或数据不一致，则进行写入
            self.save_accuracy_to_csv(accuracy_rate)
            
        except Exception as e:
            print(f"检查CSV写入需要时出错: {str(e)}")
            # 如果检查失败，直接写入
            self.save_accuracy_to_csv(accuracy_rate)
    
    def is_train_already_in_csv(self, train_name):
        """检查列车是否已在rate.csv中"""
        try:
            if not os.path.exists(self.rate_csv_path):
                return False
                
            with open(self.rate_csv_path, 'r', encoding='utf-8-sig') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row.get('列车名称', '').strip() == train_name.strip():
                        return True
            return False
        except Exception:
            return False
    
    def get_existing_accuracy_from_csv(self, train_name):
        """从现有CSV中获取准确率"""
        try:
            if not os.path.exists(self.rate_csv_path):
                return None
                
            with open(self.rate_csv_path, 'r', encoding='utf-8-sig') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row.get('列车名称', '').strip() == train_name.strip():
                        accuracy_str = row.get('准确率(%)', '0')
                        return float(accuracy_str.replace('%', ''))
            return None
        except Exception:
            return None
    
    def save_accuracy_to_csv(self, accuracy_rate):
        """保存当前列车的准确率到CSV文件"""
        try:
            # 获取当前列车名称
            current_train = self.train_combo.currentText()
            if not current_train:
                return
            
            # 获取当前列车数据
            if hasattr(self, 'current_train_data') and self.current_train_data:
                carriages = self.current_train_data.get('carriages', [])
                
                # 计算统计数据
                total_carriages = len(carriages)
                normal_carriages = [c for c in carriages if not c.get('is_special', False)]
                normal_carriages_count = len(normal_carriages)
                
                # 计算正确识别数（根据匹配类型）
                correct_count = 0
                for carriage in normal_carriages:
                    side_image_info = carriage.get('side_image_info')
                    if side_image_info:
                        match_type = side_image_info.get('match_type', 'unknown')
                        if match_type == 'exact':
                            # 完全匹配，算作正确
                            correct_count += 1
                        elif match_type == 'similar':
                            similarity_score = side_image_info.get('similarity_score', 0)
                            if similarity_score >= 4:
                                # >=4位相似，算作正确
                                correct_count += 1
                        # RFID兜底和其他情况算作错误，不增加correct_count
                
                # 保存到CSV
                self.save_train_statistics_to_csv(
                    current_train, 
                    total_carriages, 
                    correct_count, 
                    normal_carriages_count, 
                    accuracy_rate
                )
                
        except Exception as e:
            print(f"保存准确率到CSV失败: {str(e)}")

    def filter_table(self, text):
        """根据输入文本筛选表格"""
        for row in range(self.carriage_table.rowCount()):
            show_row = False
            for col in range(self.carriage_table.columnCount()):
                item = self.carriage_table.item(row, col)
                if item and text.lower() in item.text().lower():
                    show_row = True
                    break
            self.carriage_table.setRowHidden(row, not show_row)

    def on_table_row_selected(self, current_row, current_column, previous_row, previous_column):
        """当表格行被选中时"""
        if current_row < 0:
            return

        # 获取图片信息
        top_image_info = self.carriage_table.item(current_row, 0).data(Qt.UserRole)
        side_image_path = self.carriage_table.item(current_row, 2).data(Qt.UserRole)

        # 检查是否为特殊车号（P或N开头）
        carriage_number = self.carriage_table.item(current_row, 1).text().strip().upper()
        is_special = carriage_number.startswith(('P', 'N'))

        # 如果是特殊车号，不显示图片
        if is_special:
            self.clear_image()
            self.clear_side_image()
            return

        # 根据图片路径自动判断是否有水并显示相应图片
        if top_image_info and isinstance(top_image_info, dict):
            original_path = top_image_info['original_path']
            has_water = top_image_info.get('has_water', False)  # 获取积水状态
            
            # 延迟检查文件存在性（优化性能）
            if original_path:
                # 从原图路径中提取列车名和帧号
                path_parts = original_path.replace("\\", "/").split("/")
                if len(path_parts) >= 2:
                    train_folder = path_parts[-2]
                    image_name = os.path.basename(original_path)
                    frame_name = os.path.splitext(image_name)[0]  # 去掉扩展名，如frame_1
                    
                    # 根据积水状态决定显示哪张图片
                    if has_water:
                        # 如果有水，优先显示积水识别结果图片
                        water_result_image = os.path.join("runs", "output_images", train_folder, "积水识别结果", f"{frame_name}_积水识别.png")
                        
                        if os.path.exists(water_result_image):
                            # 如果积水识别结果图片存在，显示它
                            self.display_image(water_result_image)
                        else:
                            # 如果积水识别结果图片不存在，再根据CSV查找其他积水识别图片
                            has_water_csv, water_image_path = self.check_water_status_by_path(original_path)
                            if has_water_csv and water_image_path and os.path.exists(water_image_path):
                                self.display_image(water_image_path)
                            elif os.path.exists(original_path):
                                # 最后显示原始图片
                                self.display_image(original_path)
                            else:
                                self.clear_image()
                    else:
                        # 如果标记为无水，直接显示原图
                        if os.path.exists(original_path):
                            self.display_image(original_path)
                        else:
                            self.clear_image()
                else:
                    # 路径解析失败，尝试显示原始图片
                    if os.path.exists(original_path):
                        self.display_image(original_path)
                    else:
                        self.clear_image()
            else:
                self.clear_image()
        else:
            self.clear_image()

        # 显示侧面图片（延迟检查文件存在性）
        if side_image_path:
            # 检查是否为RFID兜底情况（不显示图片）
            # 通过表格数据检查匹配类型
            current_carriage_info = None
            if hasattr(self, 'current_train_data') and self.current_train_data:
                carriages = self.current_train_data.get('carriages', [])
                if current_row < len(carriages):
                    current_carriage_info = carriages[current_row]
            
            # 如果是RFID兜底情况，不显示图片
            is_rfid_fallback = False
            if current_carriage_info:
                side_image_info = current_carriage_info.get('side_image_info')
                if side_image_info and side_image_info.get('match_type') == 'rfid':
                    is_rfid_fallback = True
            
            if not is_rfid_fallback and os.path.exists(side_image_path):
                self.display_side_image(side_image_path)
            else:
                self.clear_side_image()
        else:
            self.clear_side_image()

    def check_water_status_by_path(self, original_image_path):
        """根据图片路径在CSV文件中查找积水信息（优先读取个别CSV文件）"""
        try:
            # 从图片路径中提取文件名和子文件夹
            image_name = os.path.basename(original_image_path)
            # 假设路径格式为: output_dir/train_name/image_name
            path_parts = original_image_path.replace("\\", "/").split("/")
            if len(path_parts) >= 2:
                subfolder_name = path_parts[-2]  # 子文件夹名称（即列车名）
            else:
                return False, None
            
            # 优先级1：读取各列车文件夹下的积水识别结果CSV
            individual_csv = os.path.join("runs", "output_images", subfolder_name, f"{subfolder_name}_积水识别结果.csv")
            csv_data = None
            
            if os.path.exists(individual_csv):
                print(f"优先读取个别积水识别结果文件: {individual_csv}")
                csv_data = self.cache_manager.get_csv_data(individual_csv)
            else:
                # 优先级2：回退到汇总文件
                water_summary_csv = os.path.join("runs", "output_images", "所有子文件夹积水识别结果汇总.csv")
                if os.path.exists(water_summary_csv):
                    print(f"回退到汇总文件: {water_summary_csv}")
                    csv_data = self.cache_manager.get_csv_data(water_summary_csv)
            
            if not csv_data:
                return False, None
            
            # 查找匹配记录
            for row in csv_data:
                original_name_key = '原图片名称'
                water_status_key = '是否有积水'
                detected_path_key = '识别后图片路径'
                
                # 对于个别列车CSV文件，只需匹配图片名；对于汇总文件，需要匹配子文件夹和图片名
                match_condition = False
                if os.path.exists(individual_csv):
                    # 个别CSV文件，只匹配图片名
                    match_condition = row.get(original_name_key, '').strip() == image_name
                else:
                    # 汇总文件，需要匹配子文件夹和图片名
                    subfolder_key = '子文件夹' if '子文件夹' in row else list(row.keys())[0]
                    match_condition = (row.get(subfolder_key, '').strip() == subfolder_name and 
                                     row.get(original_name_key, '').strip() == image_name)
                
                if match_condition:
                    has_water = row.get(water_status_key, '').strip() == '是'
                    water_image_path = None
                    
                    if has_water and row.get(detected_path_key, '').strip():
                        # 构建积水识别图片的完整路径
                        water_image_path = row.get(detected_path_key, '').strip()
                        # 如果是相对路径，转换为绝对路径
                        if not os.path.isabs(water_image_path):
                            water_image_path = os.path.abspath(water_image_path)
                        
                        # 优先查找积水识别结果图片
                        water_result_image = os.path.join("runs", "output_images", subfolder_name, "积水识别结果", f"{os.path.splitext(image_name)[0]}_积水识别.png")
                        if os.path.exists(water_result_image):
                            water_image_path = water_result_image
                    
                    return has_water, water_image_path
            
            return False, None
            
        except Exception as e:
            print(f"根据路径查找积水信息时出错: {str(e)}")
            return False, None

    def update_water_summary_csv(self):
        """自动更新汇总文件：将所有单独的积水识别结果汇总到一个文件中"""
        try:
            output_base_dir = os.path.join("runs", "output_images")
            summary_file = os.path.join(output_base_dir, "所有子文件夹积水识别结果汇总.csv")
            
            # 收集所有数据
            all_data = []
            
            # 遍历所有子文件夹
            if os.path.exists(output_base_dir):
                for train_folder in os.listdir(output_base_dir):
                    train_path = os.path.join(output_base_dir, train_folder)
                    if os.path.isdir(train_path) and train_folder != "所有子文件夹积水识别结果汇总.csv":
                        # 查找该列车的积水识别结果文件
                        individual_csv = os.path.join(train_path, f"{train_folder}_积水识别结果.csv")
                        if os.path.exists(individual_csv):
                            try:
                                csv_data = self.cache_manager.get_csv_data(individual_csv)
                                all_data.extend(csv_data)
                                print(f"已添加 {train_folder} 的积水识别结果到汇总文件")
                            except Exception as e:
                                print(f"读取 {individual_csv} 时出错: {str(e)}")
            
            # 写入汇总文件
            if all_data:
                import csv
                with open(summary_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
                    if all_data:
                        fieldnames = all_data[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(all_data)
                
                print(f"汇总文件已更新: {summary_file}，共 {len(all_data)} 条记录")
                
                # 更新缓存
                self.cache_manager.invalidate_csv_cache(summary_file)
                return True
            else:
                print("未找到任何积水识别结果数据")
                return False
                
        except Exception as e:
            print(f"更新汇总文件时出错: {str(e)}")
            return False

    def auto_update_summary_if_needed(self):
        """检查是否需要自动更新汇总文件"""
        try:
            output_base_dir = os.path.join("runs", "output_images")
            summary_file = os.path.join(output_base_dir, "所有子文件夹积水识别结果汇总.csv")
            
            # 检查汇总文件的修改时间
            summary_mtime = 0
            if os.path.exists(summary_file):
                summary_mtime = os.path.getmtime(summary_file)
            
            # 检查所有单独 CSV 文件的最新修改时间
            latest_individual_mtime = 0
            if os.path.exists(output_base_dir):
                for train_folder in os.listdir(output_base_dir):
                    train_path = os.path.join(output_base_dir, train_folder)
                    if os.path.isdir(train_path):
                        individual_csv = os.path.join(train_path, f"{train_folder}_积水识别结果.csv")
                        if os.path.exists(individual_csv):
                            individual_mtime = os.path.getmtime(individual_csv)
                            latest_individual_mtime = max(latest_individual_mtime, individual_mtime)
            
            # 如果有更新的单独文件，则更新汇总文件
            if latest_individual_mtime > summary_mtime:
                print("检测到新的积水识别结果，正在更新汇总文件...")
                return self.update_water_summary_csv()
            
            return False
            
        except Exception as e:
            print(f"检查汇总文件更新需求时出错: {str(e)}")
            return False
        """根据列车名称和车厢ID查找对应图片和积水信息（直接从CSV读取）"""
        try:
            # 将车厢编号转换为整数（去除前导零）
            carriage_num = int(carriage_id)

            # 构建对应的frame_X.jpg文件名
            image_name = f"frame_{carriage_num}.jpg"

            # 构建完整图片路径
            image_path = os.path.join(self.output_dir, train_name, image_name)
            
            # 查找积水识别结果汇总 CSV文件
            water_summary_csv = os.path.join("runs", "output_images", "所有子文件夹积水识别结果汇总.csv")
            has_water = False
            water_detected_image_path = None
            
            if os.path.exists(water_summary_csv):
                try:
                    with open(water_summary_csv, 'r', encoding='utf-8-sig') as csv_file:  # 使用utf-8-sig处理BOM
                        reader = csv.DictReader(csv_file)
                        # 获取列名，处理BOM问题
                        fieldnames = [name.strip('\ufeff') for name in reader.fieldnames] if reader.fieldnames else []
                        
                        for row in reader:
                            # 处理列名，对于有BOM的情况进行兼容处理
                            subfolder_key = '子文件夹' if '子文件夹' in row else list(row.keys())[0]  # 取第一列
                            original_name_key = '原图片名称'
                            water_status_key = '是否有积水'
                            detected_path_key = '识别后图片路径'
                            
                            # 检查是否匹配当前列车和图片
                            if (row.get(subfolder_key, '').strip() == train_name and 
                                row.get(original_name_key, '').strip() == image_name):
                                has_water = row.get(water_status_key, '').strip() == '是'
                                if has_water and row.get(detected_path_key, '').strip():
                                    # 构建积水识别图片的完整路径
                                    water_detected_image_path = row.get(detected_path_key, '').strip()
                                    # 如果是相对路径，转换为绝对路径
                                    if not os.path.isabs(water_detected_image_path):
                                        water_detected_image_path = os.path.abspath(water_detected_image_path)
                                break
                except Exception as e:
                    print(f"读取积水识别结果汇总CSV时出错: {str(e)}")
            
            # 返回图片信息字典
            return {
                'original_path': image_path if os.path.exists(image_path) else None,
                'water_detected_path': water_detected_image_path,
                'has_water': has_water,
                'image_name': image_name
            }
        except ValueError:
            return {
                'original_path': None,
                'water_detected_path': None,
                'has_water': False,
                'image_name': None
            }

    def find_image_for_carriage(self, train_name, carriage_id):
        """根据列车名称和车厢ID查找对应图片和积水信息"""
        try:
            # 将车厢编号转换为整数（去除前导零）
            carriage_num = int(carriage_id)

            # 构建对应的frame_X.jpg文件名
            image_name = f"frame_{carriage_num}.jpg"

            # 构建完整图片路径
            image_path = os.path.join(self.output_dir, train_name, image_name)
            
            # 查找积水识别结果汇总CSV文件
            water_summary_csv = os.path.join("runs", "output_images", "所有子文件夹积水识别结果汇总.csv")
            has_water = False
            water_detected_image_path = None
            
            if os.path.exists(water_summary_csv):
                try:
                    with open(water_summary_csv, 'r', encoding='utf-8') as csv_file:
                        reader = csv.DictReader(csv_file)
                        for row in reader:
                            # 检查是否匹配当前列车和图片
                            if (row['子文件夹'] == train_name and 
                                row['原图片名称'] == image_name):
                                has_water = row['是否有积水'] == '是'
                                if has_water:
                                    # 如果有积水，使用识别后的图片路径
                                    water_detected_image_path = row['识别后图片路径']
                                break
                except Exception as e:
                    print(f"读取积水识别结果汇总CSV时出错: {str(e)}")
            
            # 返回图片信息字典
            return {
                'original_path': image_path if os.path.exists(image_path) else None,
                'water_detected_path': water_detected_image_path,
                'has_water': has_water,
                'image_name': image_name
            }
        except ValueError:
            return {
                'original_path': None,
                'water_detected_path': None,
                'has_water': False,
                'image_name': None
            }

    def find_side_image_by_plate_number(self, train_name, full_carriage_number):
        """根据完整车号查找匹配的侧面图片和识别结果（仅在同名文件夹中）"""
        try:
            # 构建CSV文件路径（仅查找同名文件夹）
            csv_path = os.path.join("output_images", train_name, "plate_results.csv")
            if not os.path.exists(csv_path):
                return None

            # 读取CSV文件
            with open(csv_path, 'r') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    # 比较车号，忽略大小写和空格
                    if row['plate_number'].strip().upper() == full_carriage_number.strip().upper():
                        return {
                            'image_path': os.path.join("output_images", train_name, row['image_name']),
                            'plate_number': row['plate_number'],
                            'timestamp': row['timestamp']
                        }
            return None
        except Exception as e:
            print(f"查找侧面图片信息时出错: {str(e)}")
            return None

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

    def display_image(self, image_path):
        """显示顶部图片"""
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            # 创建新的QLabel以适应图片大小
            self.image_label = QLabel()
            self.image_label.setAlignment(Qt.AlignCenter)

            # 调整图片大小，适应窗口但不超过原始大小的90%
            max_width = self.image_scroll_area.width() * 0.9
            max_height = self.image_scroll_area.height() * 0.9

            # 如果图片比设定的最大尺寸大，则缩小
            if pixmap.width() > max_width or pixmap.height() > max_height:
                pixmap = pixmap.scaled(
                    int(max_width),
                    int(max_height),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )

            self.image_label.setPixmap(pixmap)
            self.image_scroll_area.setWidget(self.image_label)
        else:
            self.clear_image()

    def display_side_image(self, image_path):
        """显示侧面图片"""
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            # 创建新的QLabel以适应图片大小
            self.side_image_label = QLabel()
            self.side_image_label.setAlignment(Qt.AlignCenter)

            # 调整图片大小，适应窗口但不超过原始大小的90%
            max_width = self.side_image_scroll_area.width() * 0.9
            max_height = self.side_image_scroll_area.height() * 0.9

            # 如果图片比设定的最大尺寸大，则缩小
            if pixmap.width() > max_width or pixmap.height() > max_height:
                pixmap = pixmap.scaled(
                    int(max_width),
                    int(max_height),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )

            self.side_image_label.setPixmap(pixmap)
            self.side_image_scroll_area.setWidget(self.side_image_label)
        else:
            self.clear_side_image()

    def clear_image(self):
        """清除顶部图片显示"""
        self.image_label = QLabel("无图片")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            font-size: 16px; 
            color: #95a5a6;
            background-color: white;
            padding: 30px;
        """)
        self.image_scroll_area.setWidget(self.image_label)

    def clear_side_image(self):
        """清除侧面图片显示"""
        self.side_image_label = QLabel("无侧面图片")
        self.side_image_label.setAlignment(Qt.AlignCenter)
        self.side_image_label.setStyleSheet("""
            font-size: 16px; 
            color: #95a5a6;
            background-color: white;
            padding: 30px;
        """)
        self.side_image_scroll_area.setWidget(self.side_image_label)

    def refresh_with_cache_clear(self):
        """刷新列车文件并清空缓存"""
        # 清空缓存
        self.cache_manager.invalidate_cache()
        # 重新加载列车文件
        self.load_train_files()
    
    def auto_process_all_trains_to_csv(self):
        """自动处理所有列车信息并写入rate.csv（增量更新，只处理新增列车）"""
        try:
            print("开始增量更新rate.csv...")
            
            # 获取所有列车文件
            total_count = self.train_combo.count()
            if total_count == 0:
                print("没有找到列车文件")
                return
            
            # 获取已存在于rate.csv中的列车名称
            existing_trains = self.get_existing_trains_from_csv()
            print(f"CSV中已存在 {len(existing_trains)} 趟列车的记录")
            
            # 筛选出需要处理的新列车
            trains_to_process = []
            for i in range(total_count):
                train_name = self.train_combo.itemText(i)
                file_path = self.train_combo.itemData(i)
                
                if train_name and train_name not in existing_trains:
                    trains_to_process.append((train_name, file_path))
            
            if not trains_to_process:
                print("所有列车数据已存在于rate.csv中，无需处理")
                return
            
            print(f"发现 {len(trains_to_process)} 趟新列车需要处理")
            processed_count = 0
            
            # 只处理新列车
            for train_name, file_path in trains_to_process:
                if not train_name or not file_path or not os.path.exists(file_path):
                    continue
                
                try:
                    print(f"正在处理新列车: {train_name}")
                    
                    # 加载列车数据（使用缓存）
                    train_data = self.cache_manager.get_train_data(
                        train_name, 
                        file_path, 
                        self.load_train_data_with_cache
                    )
                    
                    if train_data:
                        # 计算统计数据
                        carriages = train_data.get('carriages', [])
                        normal_carriages = [c for c in carriages if not c.get('is_special', False)]
                        
                        total_carriages = len(carriages)
                        normal_carriages_count = len(normal_carriages)
                        accuracy_rate = train_data.get('accuracy_rate', 0.0)
                        
                        # 计算正确识别数
                        correct_count = 0
                        for carriage in normal_carriages:
                            side_image_info = carriage.get('side_image_info')
                            if side_image_info:
                                match_type = side_image_info.get('match_type', 'unknown')
                                if match_type == 'exact':
                                    correct_count += 1
                                elif match_type == 'similar':
                                    similarity_score = side_image_info.get('similarity_score', 0)
                                    if similarity_score >= 5:
                                        correct_count += 1
                        
                        # 写入CSV（追加模式）
                        self.append_train_to_csv(
                            train_name, 
                            total_carriages, 
                            correct_count, 
                            normal_carriages_count, 
                            accuracy_rate
                        )
                        
                        processed_count += 1
                        print(f"✓ 已处理: {train_name} (进度: {processed_count}/{len(trains_to_process)})")
                        
                except Exception as e:
                    print(f"✗ 处理列车 {train_name} 时出错: {str(e)}")
                    continue
            
            print(f"增量更新完成！新增 {processed_count} 趟列车的记录")
            
        except Exception as e:
            print(f"增量更新rate.csv时出错: {str(e)}")
    
    def get_existing_trains_from_csv(self):
        """获取rate.csv中已存在的列车名称列表"""
        existing_trains = set()
        
        try:
            if os.path.exists(self.rate_csv_path):
                with open(self.rate_csv_path, 'r', encoding='utf-8-sig') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        train_name = row.get('列车名称', '').strip()
                        if train_name:
                            existing_trains.add(train_name)
                            
                print(f"从CSV中读取到 {len(existing_trains)} 趟已存在的列车")
            else:
                print("rate.csv文件不存在，将创建新文件")
                
        except Exception as e:
            print(f"读取现有CSV记录时出错: {str(e)}")
            
        return existing_trains
    
    def append_train_to_csv(self, train_name, total_carriages, correct_count, normal_carriages_count, accuracy_rate):
        """将新列车记录追加到rate.csv文件（高效追加模式）"""
        try:
            # 确保目录存在
            os.makedirs(self.data_dir, exist_ok=True)
            
            # 检查文件是否存在，如果不存在则写入表头
            file_exists = os.path.exists(self.rate_csv_path)
            
            # 准备新记录
            new_record = {
                '列车名称': train_name,
                '总车厢数': total_carriages,
                '识别正确数': correct_count,
                '参与统计数': normal_carriages_count,
                '准确率(%)': f"{accuracy_rate:.2f}",
                '统计时间': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 追加写入文件
            with open(self.rate_csv_path, 'a', newline='', encoding='utf-8-sig') as file:
                fieldnames = ['列车名称', '总车厢数', '识别正确数', '参与统计数', '准确率(%)', '统计时间']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                
                # 如果是新文件，写入表头
                if not file_exists:
                    writer.writeheader()
                    print(f"创建新的rate.csv文件: {self.rate_csv_path}")
                
                # 写入新记录
                writer.writerow(new_record)
                
            print(f"已追加列车记录: {train_name} -> {self.rate_csv_path}")
            
        except Exception as e:
            print(f"追加列车记录失败: {str(e)}")
    
    def smart_update_csv_in_background(self):
        """智能后台更新（非阻塞，只处理新增数据）"""
        try:
            # 获取需要处理的新列车数量
            existing_trains = self.get_existing_trains_from_csv()
            total_trains = self.train_combo.count()
            
            new_trains_count = 0
            for i in range(total_trains):
                train_name = self.train_combo.itemText(i)
                if train_name and train_name not in existing_trains:
                    new_trains_count += 1
            
            if new_trains_count == 0:
                print("✓ 所有列车数据已存在于rate.csv中")
                return
            
            print(f"发现 {new_trains_count} 趟新列车，将在后台进行增量更新...")
            
            # 启动后台更新（非阻塞）
            import threading
            
            def background_update():
                self.auto_process_all_trains_to_csv()
                print("✓ 后台增量更新完成！")
            
            # 创建后台线程
            update_thread = threading.Thread(target=background_update, daemon=True)
            update_thread.start()
            
        except Exception as e:
            print(f"智能后台更新出错: {str(e)}")
    
    def save_accuracy_to_csv(self, accuracy_rate):
        """保存当前列车的准确率到CSV文件（界面调用）"""
        try:
            # 获取当前列车名称
            current_train = self.train_combo.currentText()
            if not current_train:
                return
            
            # 获取当前列车数据
            if hasattr(self, 'current_train_data') and self.current_train_data:
                carriages = self.current_train_data.get('carriages', [])
                
                # 计算统计数据
                total_carriages = len(carriages)
                normal_carriages = [c for c in carriages if not c.get('is_special', False)]
                normal_carriages_count = len(normal_carriages)
                
                # 计算正确识别数（根据匹配类型）
                correct_count = 0
                for carriage in normal_carriages:
                    side_image_info = carriage.get('side_image_info')
                    if side_image_info:
                        match_type = side_image_info.get('match_type', 'unknown')
                        if match_type == 'exact':
                            # 完全匹配，算作正确
                            correct_count += 1
                        elif match_type == 'similar':
                            similarity_score = side_image_info.get('similarity_score', 0)
                            if similarity_score >= 4:
                                # >=4位相似，算作正确
                                correct_count += 1
                        # RFID兜底和其他情况算作错误，不增加correct_count
                
                # 保存到CSV
                self.save_train_statistics_to_csv(
                    current_train, 
                    total_carriages, 
                    correct_count, 
                    normal_carriages_count, 
                    accuracy_rate
                )
                
        except Exception as e:
            print(f"保存准确率到CSV失败: {str(e)}")



class WaterDetectionDialog(QDialog):
    """积水识别模式选择对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("积水识别系统")
        self.setFixedSize(400, 300)
        self.selected_mode = None
        
        # 设置样式
        self.setStyleSheet("""
            QDialog {
                background-color: white;
            }
            QLabel {
                color: #000000;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1c6ea4;
            }
        """)
        
        self.init_ui()
        self.center_window()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题
        title = QLabel("积水识别系统")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #000000; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 说明文本
        description = QLabel("请选择积水识别的工作模式：")
        description.setAlignment(Qt.AlignCenter)
        layout.addWidget(description)
        
        # 按钮区域
        button_layout = QVBoxLayout()
        button_layout.setSpacing(15)
        
        # 监控模式按钮
        monitor_btn = QPushButton("📹 启动监控模式")
        monitor_btn.setStyleSheet("background-color: #27ae60; min-height: 40px;")
        monitor_btn.clicked.connect(lambda: self.select_mode("monitor"))
        button_layout.addWidget(monitor_btn)
        
        monitor_desc = QLabel("自动监控文件夹，检测到新图片时自动进行积水识别")
        monitor_desc.setStyleSheet("font-size: 12px; color: #000000; margin-bottom: 10px;")
        monitor_desc.setAlignment(Qt.AlignCenter)
        button_layout.addWidget(monitor_desc)
        
        # 批量处理模式按钮
        batch_btn = QPushButton("📁 启动批量处理")
        batch_btn.setStyleSheet("background-color: #e74c3c; min-height: 40px;")
        batch_btn.clicked.connect(lambda: self.select_mode("batch"))
        button_layout.addWidget(batch_btn)
        
        batch_desc = QLabel("一次性处理所有现有文件夹中的图片")
        batch_desc.setStyleSheet("font-size: 12px; color: #000000; margin-bottom: 20px;")
        batch_desc.setAlignment(Qt.AlignCenter)
        button_layout.addWidget(batch_desc)
        
        layout.addLayout(button_layout)
        
        # 取消按钮
        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet("background-color: #95a5a6;")
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)
    
    def select_mode(self, mode):
        """选择模式"""
        self.selected_mode = mode
        self.accept()
    
    def get_selected_mode(self):
        """获取选中的模式"""
        return self.selected_mode
    
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

    def open_train_info(self):
        """打开列车信息查看窗口"""
        try:
            # 导入列车信息查看器
            from train_info_viewer import TrainInfoViewer
            
            # 创建列车信息查看窗口
            self.train_info_window = TrainInfoViewer(self)
            self.train_info_window.show()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开列车信息查看器失败：{str(e)}")




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    # 确保窗口大小被正确计算后再居中显示
    window.adjustSize()
    window.center_window()
    window.show()
    sys.exit(app.exec())
