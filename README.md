# EmotiSense: Real-time Emotion Detection System

A real-time emotion detection system powered by OpenCV and DeepFace, capable of analyzing facial expressions and emotional states through webcam. Features include real-time face detection, emotion tracking, high-confidence emotion logging (>95%), and emotion trend analysis with DeepSeek API integration.

# 实时情绪检测系统

这是一个基于OpenCV和DeepFace的实时情绪检测系统，可以通过摄像头实时检测人脸表情并分析情绪状态。

## 功能特点

- 实时人脸检测和眼睛追踪
- 情绪状态实时分析
- 高强度情绪记录（置信度>95%）
- 情绪数据日志记录
- DeepSeek API集成的情绪趋势分析

## 环境要求

- Python 3.7+
- OpenCV
- DeepFace
- TensorFlow

## 依赖安装

```bash
pip install -r requirements.txt
```

## 配置说明

1. 创建`.env`文件并配置DeepSeek API密钥：
```
DEEPSEEK_API_KEY=your_api_key_here
```

## 使用方法

1. 运行程序：
```bash
python main.py
```

2. 操作说明：
- 程序启动后会自动打开摄像头
- 实时显示检测到的人脸和情绪状态
- 按'q'或'ESC'键退出程序

## 输出说明

- 实时显示：显示摄像头画面，包含人脸框、眼睛框和情绪状态
- 日志文件：`emotion_log.txt`记录高强度情绪状态（>95%置信度）
- 情绪分析：程序结束时会对收集的情绪数据进行分析

## 性能优化

- 降低分辨率（640x360）以减少处理负担
- 帧跳过机制（每2帧处理一次）
- 定期清理旧数据以控制内存使用
- 人脸框位置平滑处理，提供更稳定的显示效果

## 技术实现

### 1. 情绪检测引擎
- 使用DeepFace库作为核心情绪分析引擎
  - 预训练模型支持7种基本情绪识别
  - 实时分析每帧图像中的面部表情
  - 输出情绪类型和置信度

### 2. 视频处理流程
- 基于OpenCV实现实时视频捕获和处理
  - 使用cv2.VideoCapture获取摄像头数据流
  - 应用Haar级联分类器进行人脸和眼睛检测
  - 图像预处理：缩放、灰度转换、直方图均衡化

### 3. 数据分析与API集成
- 情绪数据收集和存储
  - 实时记录高置信度（>95%）的情绪状态
  - 使用时间序列存储情绪变化趋势
- DeepSeek API集成
  - 定期发送情绪数据进行深度分析
  - 生成个性化情绪趋势报告

### 4. 性能优化策略
- 图像处理优化
  - 降低分辨率（640x360）
  - 帧跳过处理（每2帧分析一次）
  - ROI区域动态调整
- 内存管理
  - 定期清理历史数据
  - 限制情绪记录数量（最大1000条）
  - 采用数据流式处理减少内存占用

## 内存占用

- 图像处理：
  - 摄像头帧缓存：约 0.7MB (640x360 分辨率)
  - 人脸ROI区域：约 0.1MB
  - 灰度图像：约 0.2MB

- 模型加载：
  - DeepFace模型：约 100MB
  - 人脸检测级联分类器：约 2MB
  - 眼睛检测级联分类器：约 1MB

- 数据存储：
  - 情绪数据列表：最大 1000条记录，约 0.5MB
  - 日志文件：根据使用时长动态增长

总体运行内存占用：约 150-200MB

## 注意事项

1. 确保摄像头可正常使用
2. 保持适当的光照条件以提高检测准确度
3. 建议在相对安静的环境中使用，以获得更准确的情绪分析结果
