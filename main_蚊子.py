#
# file: requirements.txt
#
# 运行此完整系统所需的Python包。
#
# 您可以使用以下命令进行安装:
# pip install -r requirements.txt
#

pyqt5
pyqtgraph
numpy
opencv-python
pygame

```python
#
# file: mosquito_radar_system_integrated.py
#
# 蚊虫雷达追踪与威慑系统 - 完整功能集成版
#
# 该脚本整合了研究报告中描述的所有核心功能：
# 1. 视觉系统 (第一章): 使用OpenCV进行实时蚊虫检测与追踪。
# 2. 威慑系统 (第二章): 使用PyGame生成并播放超声波。
# 3. 人机界面 (第三章): 使用PyQtGraph进行高性能可视化。
#
# 运行说明:
# 1. 确保已安装 requirements.txt 中的所有库。
# 2. 直接运行此脚本: python mosquito_radar_system_integrated.py
# 3. 按 'Q' 退出, 'A' 手动切换声波, 'P' 截图。


import sys
import time
import numpy as np
import cv2
import pygame
import wave
import io
import math
from collections import deque

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QFrame, QGridLayout)
from PyQt5.QtGui import QFont, QColor, QPixmap
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
import pyqtgraph as pg

# --- 全局样式和配置 (来自HMI设计) ---
BACKGROUND_COLOR = "#000000"
PRIMARY_COLOR = "#00FF41"
ACCENT_COLOR = "#FF0000"
PULSE_COLOR = "#FFFF00"
FONT_NAME = "Consolas"

# --- 视觉系统配置 (来自第一章) ---
CAM_INDEX = 0  # 摄像头索引，0通常是默认摄像头
MIN_AREA = 5          # 蚊虫最小面积 (px²)
MAX_AREA = 150        # 蚊虫最大面积 (px²)
MIN_PERIMETER = 10    # 蚊虫最小周长 (px)
MIN_CIRCULARITY = 0.5 # 蚊虫最小圆形度
EDGE_THRESHOLD1 = 100 # Canny边缘检测阈值1
EDGE_THRESHOLD2 = 200 # Canny边缘检测阈值2
MAX_TRACK_AGE = 30    # 目标失追前的最大帧数
MAX_TRAIL_LENGTH = 50 # 轨迹最大长度

# --- 威慑系统配置 (来自第二章) ---
ULTRASOUND_FREQ = 22000  # 超声波频率 (Hz)
ULTRASOUND_DURATION = 1  # WAV文件时长 (秒)
ULTRASOUND_VOLUME = 1.0  # 音量 (0.0 to 1.0)
AUTO_ATTACK_DELAY = 5    # 自动攻击间隔 (秒)

# ==============================================================================
# 视觉追踪线程 (第一章实现)
# ==============================================================================
class VisionThread(QThread):
    """
    在独立线程中处理所有计算机视觉任务，以防止GUI冻结。
    """
    targets_updated = pyqtSignal(dict)  # 发送追踪到的目标数据的信号
    camera_status = pyqtSignal(str)     # 发送摄像头状态的信号

    def __init__(self):
        super().__init__()
        self.running = True
        self.next_target_id = 0
        self.tracked_targets = {}

        # MOG2背景减法器
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    def run(self):
        """线程主循环"""
        cap = cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened():
            self.camera_status.emit("连接失败")
            return
        self.camera_status.emit("已连接")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.camera_status.emit("读取失败")
                time.sleep(1)
                continue

            # 1. 图像预处理
            processed_frame = self.process_frame(frame)

            # 2. 轮廓检测与过滤
            contours = self.find_contours(processed_frame)
            filtered_contours = self.filter_contours(contours)

            # 3. 目标追踪
            self.update_tracking(filtered_contours)

            # 4. 发送数据到主线程
            self.targets_updated.emit(self.tracked_targets)

            time.sleep(1/30) # 控制帧率

        cap.release()

    def process_frame(self, frame):
        """应用背景减除和边缘检测"""
        fgMask = self.backSub.apply(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, EDGE_THRESHOLD1, EDGE_THRESHOLD2)
        # 结合运动和边缘信息
        combined = cv2.bitwise_and(fgMask, edges)
        return combined

    def find_contours(self, frame):
        """从处理后的帧中找到轮廓"""
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def filter_contours(self, contours):
        """根据特征过滤轮廓，仅保留可能是蚊子的"""
        filtered = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if MIN_AREA < area < MAX_AREA and perimeter > MIN_PERIMETER:
                if perimeter > 0:
                    circularity = 4 * math.pi * (area / (perimeter * perimeter))
                    if circularity > MIN_CIRCULARITY:
                        filtered.append(cnt)
        return filtered

    def update_tracking(self, contours):
        """基于轨迹匹配的多目标追踪"""
        detected_centers = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                detected_centers.append(np.array([cx, cy]))

        # 标记所有目标为未匹配
        for tid in self.tracked_targets:
            self.tracked_targets[tid]['matched'] = False

        # 尝试将新检测与现有目标匹配
        for center in detected_centers:
            best_match_id = -1
            min_dist = 50  # 最大匹配距离

            for tid, data in self.tracked_targets.items():
                dist = np.linalg.norm(center - data['pos'])
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = tid

            if best_match_id != -1:
                # 匹配成功，更新目标
                self.tracked_targets[best_match_id]['pos'] = center
                self.tracked_targets[best_match_id]['trail'].append(center)
                self.tracked_targets[best_match_id]['age'] = 0
                self.tracked_targets[best_match_id]['matched'] = True
            else:
                # 未找到匹配，创建新目标
                self.tracked_targets[self.next_target_id] = {
                    'pos': center,
                    'trail': deque([center], maxlen=MAX_TRAIL_LENGTH),
                    'age': 0,
                    'matched': True
                }
                self.next_target_id += 1

        # 清理失追的目标
        lost_ids = []
        for tid, data in self.tracked_targets.items():
            if not data['matched']:
                data['age'] += 1
            if data['age'] > MAX_TRACK_AGE:
                lost_ids.append(tid)

        for tid in lost_ids:
            del self.tracked_targets[tid]

    def stop(self):
        self.running = False

# ==============================================================================
# 超声波威慑系统 (第二章实现)
# ==============================================================================
class DeterrenceSystem:
    """管理超声波的生成和播放"""
    def __init__(self):
        pygame.mixer.init(frequency=ULTRASOUND_FREQ, channels=1)
        self.sound_data = self._generate_wav_data()
        self.sound = pygame.mixer.Sound(buffer=self.sound_data)
        self.is_active = False
        self.last_attack_time = 0

    def _generate_wav_data(self):
        """在内存中生成WAV文件数据"""
        sample_rate = ULTRASOUND_FREQ
        n_samples = int(ULTRASOUND_DURATION * sample_rate)
        n_channels = 1
        sampwidth = 2
        
        wav_data = io.BytesIO()
        with wave.open(wav_data, 'wb') as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)

            max_amplitude = 32767  # 16-bit
            for i in range(n_samples):
                value = int(max_amplitude * math.sin(2 * math.pi * ULTRASOUND_FREQ * i / sample_rate))
                wf.writeframes(np.int16(value).tobytes())
        
        wav_data.seek(0)
        return wav_data.read()

    def set_active(self, active):
        """设置威慑系统状态"""
        if active and not self.is_active:
            self.sound.play(loops=-1, fade_ms=500)
            self.is_active = True
        elif not active and self.is_active:
            self.sound.fadeout(500)
            self.is_active = False

    def toggle(self):
        """切换威慑系统开关"""
        self.set_active(not self.is_active)

    def auto_attack(self, has_targets):
        """自动攻击逻辑"""
        current_time = time.time()
        if has_targets:
            if not self.is_active:
                self.set_active(True)
            self.last_attack_time = current_time
        elif self.is_active and (current_time - self.last_attack_time > AUTO_ATTACK_DELAY):
            self.set_active(False)

# ==============================================================================
# 主窗口和HMI (第三章实现)
# ==============================================================================
class RadarWindow(QMainWindow):
    """主窗口类，整合所有功能"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("蚊虫雷达追踪与威慑系统 - 完整版")
        self.setGeometry(100, 100, 1600, 900)

        # --- 初始化系统模块 ---
        self.deterrence = DeterrenceSystem()
        self.deterrence_manual_mode = False

        # --- 状态变量 ---
        self.start_time = time.time()
        self.targets = {}
        self.scan_angle = 0
        self.pulse_phase = 0
        self.total_detections_today = 0
        self.max_concurrent_targets = 0
        self.accuracy = 98.5 # 模拟值
        self.false_positive_rate = 1.5 # 模拟值

        self.setup_ui()

        # --- 视觉线程 ---
        self.vision_thread = VisionThread()
        self.vision_thread.targets_updated.connect(self.on_targets_updated)
        self.vision_thread.camera_status.connect(self.on_camera_status_update)
        self.vision_thread.start()

        # --- UI更新定时器 ---
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(30) # ~33 FPS

    def setup_ui(self):
        """设置用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.setStyleSheet(f"background-color: {BACKGROUND_COLOR}; color: {PRIMARY_COLOR};")
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        self.setup_radar_plot()
        main_layout.addWidget(self.plot_widget, 7)

        self.setup_info_panel()
        main_layout.addWidget(self.info_panel, 3)

    def setup_radar_plot(self):
        """初始化雷达图"""
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(BACKGROUND_COLOR)
        self.plot_widget.showGrid(x=False, y=False)
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.getPlotItem().hideAxis('bottom')
        self.plot_widget.getPlotItem().hideAxis('left')

        view_range = 500
        self.plot_widget.setXRange(-view_range, view_range)
        self.plot_widget.setYRange(-view_range, view_range)

        grid_color = QColor(PRIMARY_COLOR)
        grid_color.setAlpha(80)
        pen = pg.mkPen(color=grid_color, style=Qt.DotLine)
        for r in range(100, 600, 100):
            circle = pg.QtGui.QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
            circle.setPen(pen)
            self.plot_widget.addItem(circle)

        self.scan_line = self.plot_widget.plot(pen=pg.mkPen(color=PRIMARY_COLOR, width=2))
        self.target_scatter = pg.ScatterPlotItem(pen=pg.mkPen(None))
        self.trail_scatter = pg.ScatterPlotItem(pen=pg.mkPen(None))
        self.plot_widget.addItem(self.target_scatter)
        self.plot_widget.addItem(self.trail_scatter)

    def setup_info_panel(self):
        """初始化信息面板"""
        self.info_panel = QWidget()
        panel_layout = QVBoxLayout(self.info_panel)
        panel_layout.setSpacing(15)

        def create_info_group(title):
            group_frame = QFrame()
            group_frame.setFrameShape(QFrame.StyledPanel)
            group_frame.setStyleSheet(f"border: 1px solid {PRIMARY_COLOR}; border-radius: 5px; margin-top: 1em;")
            layout = QGridLayout(group_frame)
            title_label = QLabel(title)
            title_label.setFont(QFont(FONT_NAME, 12, QFont.Bold))
            title_label.setStyleSheet(f"margin-left: 10px; margin-top: -0.8em; background: {BACKGROUND_COLOR}; padding: 0 5px;")
            layout.addWidget(title_label, 0, 0, 1, 2)
            return group_frame, layout

        self.labels = {}
        groups_data = {
            "SYSTEM STATUS": {
                "运行时间": "00:00:00", "摄像头状态": "初始化...", "追踪精度": f"{self.accuracy}%", "误报率": f"{self.false_positive_rate}%"
            },
            "DETECTION STATS": {
                "当前蚊虫数": "0", "今日累计": "0", "最大同时数量": "0"
            },
            "DETERRENCE SYSTEM": {
                "模式": "自动", "状态": "待命中"
            },
            "CONTROLS": {
                "快捷键": "Q: 退出 | A: 切换模式 | P: 截图"
            }
        }

        for title, data in groups_data.items():
            group, layout = create_info_group(title)
            row = 1
            for key, value in data.items():
                key_label = QLabel(f"{key}:")
                key_label.setFont(QFont(FONT_NAME, 10))
                value_label = QLabel(value)
                value_label.setFont(QFont(FONT_NAME, 10, QFont.Bold))
                self.labels[key] = value_label
                layout.addWidget(key_label, row, 0)
                layout.addWidget(value_label, row, 1)
                row += 1
            panel_layout.addWidget(group)

        panel_layout.addStretch()

    def on_targets_updated(self, targets_from_thread):
        """当视觉线程发来新数据时调用"""
        if len(targets_from_thread) > len(self.targets):
             self.total_detections_today += len(targets_from_thread) - len(self.targets)
        
        self.targets = targets_from_thread
        
        if not self.deterrence_manual_mode:
            self.deterrence.auto_attack(len(self.targets) > 0)

    def on_camera_status_update(self, status):
        """更新摄像头状态标签"""
        self.labels["摄像头状态"].setText(status)

    def update_ui(self):
        """主UI更新循环"""
        self.update_radar_plot()
        self.update_info_panel()

    def update_radar_plot(self):
        """更新雷达图"""
        self.scan_angle = (self.scan_angle + 5) % 360
        rad = np.deg2rad(self.scan_angle)
        self.scan_line.setData([0, 500 * np.cos(rad)], [0, 500 * np.sin(rad)])

        target_points = []
        trail_points = []
        trail_brushes = []

        self.pulse_phase = (self.pulse_phase + 0.2) % (2 * np.pi)
        pulse_size = 15 + 5 * np.sin(self.pulse_phase)
        
        is_attacking = self.deterrence.is_active

        for tid, t_data in self.targets.items():
            # 坐标转换：OpenCV左上角(0,0) -> 雷达中心(0,0)
            # 假设摄像头分辨率为640x480，雷达范围为-500到500
            x = t_data['pos'][0] - 320 
            y = -(t_data['pos'][1] - 240)
            
            size = pulse_size if is_attacking else 10
            brush = pg.mkBrush(PULSE_COLOR) if is_attacking else pg.mkBrush(ACCENT_COLOR)
            target_points.append({'pos': (x, y), 'size': size, 'brush': brush})

            for i, point in enumerate(list(t_data['trail'])):
                trail_x = point[0] - 320
                trail_y = -(point[1] - 240)
                trail_points.append((trail_x, trail_y))
                alpha = int(150 * (i / MAX_TRAIL_LENGTH)) # 基于索引计算透明度
                trail_brushes.append(pg.mkBrush(color=(0, 255, 65, alpha)))

        self.target_scatter.setData(target_points)
        self.trail_scatter.setData(pos=np.array(trail_points), brush=trail_brushes)

    def update_info_panel(self):
        """更新信息面板"""
        elapsed = time.time() - self.start_time
        self.labels["运行时间"].setText(time.strftime("%H:%M:%S", time.gmtime(elapsed)))

        num_targets = len(self.targets)
        if num_targets > self.max_concurrent_targets:
            self.max_concurrent_targets = num_targets

        self.labels["当前蚊虫数"].setText(str(num_targets))
        self.labels["今日累计"].setText(str(self.total_detections_today))
        self.labels["最大同时数量"].setText(str(self.max_concurrent_targets))
        
        mode_text = "手动" if self.deterrence_manual_mode else "自动"
        status_text = "开启" if self.deterrence.is_active else "关闭"
        self.labels["模式"].setText(mode_text)
        self.labels["状态"].setText(status_text)
        self.labels["状态"].setStyleSheet(f"color: {ACCENT_COLOR if self.deterrence.is_active else PRIMARY_COLOR};")

    def keyPressEvent(self, event):
        """处理快捷键"""
        key = event.key()
        if key == Qt.Key_Q:
            self.close()
        elif key == Qt.Key_A:
            self.deterrence_manual_mode = not self.deterrence_manual_mode
            if self.deterrence_manual_mode:
                self.deterrence.toggle()
        elif key == Qt.Key_P:
            self.screenshot()

    def screenshot(self):
        """截取当前窗口画面"""
        screen = QApplication.primaryScreen()
        filename = f"mosquito_radar_{time.strftime('%Y%m%d_%H%M%S')}.png"
        screenshot = screen.grabWindow(self.winId())
        screenshot.save(filename, 'png')
        print(f"截图已保存为: {filename}")

    def closeEvent(self, event):
        """关闭窗口时清理资源"""
        self.vision_thread.stop()
        self.vision_thread.wait()
        self.deterrence.set_active(False)
        pygame.quit()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RadarWindow()
    window.show()
    sys.exit(app.exec_())
