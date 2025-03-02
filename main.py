import cv2
import time
from deepface import DeepFace
import requests
import json
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化DeepFace模型，避免重复加载
emotion_model = None

# 设置情绪数据存储的最大数量
MAX_EMOTION_DATA = 1000

def analyze_emotion(face_img):
    """使用DeepFace进行情绪分析，返回最可能的情绪"""
    global emotion_model
    try:
        # 延迟加载模型
        if emotion_model is None:
            result = DeepFace.analyze(face_img, 
                                    actions=['emotion'],
                                    enforce_detection=False)
            emotion_model = result[0]['emotion']
        else:
            result = DeepFace.analyze(face_img, 
                                    actions=['emotion'],
                                    enforce_detection=False,
                                    detector_backend='skip')
        
        # 获取情绪分析结果
        emotions = result[0]['emotion']
        
        # 提高anger的阈值，只有当愤怒值超过50%时才会被识别为愤怒
        if emotions['angry'] < 50:
            emotions['angry'] = 0
            
        # 返回最可能的情绪
        top_emotion = max(emotions.items(), key=lambda x: x[1])
        return top_emotion
    except Exception as e:
        print(f'情绪分析错误: {e}')
        return ('Unknown', 0)

def clean_old_data(emotion_data, max_size=MAX_EMOTION_DATA):
    """清理旧的情绪数据，只保留最新的max_size条记录"""
    if len(emotion_data) > max_size:
        # 使用切片创建新列表而不是原地修改
        emotion_data = emotion_data[-max_size:]
    # 手动触发垃圾回收
    import gc
    gc.collect()
    return emotion_data

def main():
    # 初始化摄像头并设置较低的分辨率以减少内存占用
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 降低分辨率
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)  # 提高帧率以使显示更流畅
    
    # 加载人脸和眼睛检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # 创建窗口
    cv2.namedWindow('情绪检测', cv2.WINDOW_NORMAL)
    
    # 初始化情绪数据收集列表和检测参数
    emotion_data = []
    last_detection_time = time.time()
    last_cleanup_time = time.time()
    detection_interval = 3.0  # 提高检测频率到每3秒一次
    cleanup_interval = 30  # 每30秒清理一次旧数据
    frame_skip = 2  # 每2帧处理一次
    frame_count = 0
    
    # 初始化上一帧的人脸位置
    last_face_rect = None
    smoothing_factor = 0.3  # 平滑因子，用于框的位置平滑
    
    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print('无法读取摄像头')
            break
            
        current_time = time.time()
        frame_count += 1
        
        # 跳过部分帧以减少处理负担
        if frame_count % frame_skip != 0:
            # 仅显示视频窗口
            cv2.imshow('情绪检测', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27是ESC键的ASCII码
                break
            continue
        
        # 定期清理旧数据
        if current_time - last_cleanup_time >= cleanup_interval:
            emotion_data = clean_old_data(emotion_data)
            last_cleanup_time = current_time
        
        if current_time - last_detection_time < detection_interval:
            continue
        
        last_detection_time = current_time
        
        # 保持较低的显示分辨率
        frame = cv2.resize(frame, (640, 360))
        
        # 转换为灰度图像进行人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(80, 80), maxSize=(300, 300))
        
        # 如果检测到人脸，更新位置
        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # 只处理第一个检测到的人脸
            
            # 如果存在上一帧的人脸位置，进行平滑处理
            if last_face_rect is not None:
                x = int(last_face_rect[0] * (1 - smoothing_factor) + x * smoothing_factor)
                y = int(last_face_rect[1] * (1 - smoothing_factor) + y * smoothing_factor)
                w = int(last_face_rect[2] * (1 - smoothing_factor) + w * smoothing_factor)
                h = int(last_face_rect[3] * (1 - smoothing_factor) + h * smoothing_factor)
            
            # 更新上一帧的人脸位置
            last_face_rect = (x, y, w, h)
            # 绘制人脸框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # 在人脸区域内检测眼睛
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(20, 20))
            
            # 绘制眼睛框
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # 分析情绪
            face_img = frame[y:y+h, x:x+w].copy()  # 创建副本以避免内存泄漏
            emotion, prob = analyze_emotion(face_img)
            del face_img  # 及时释放不再需要的图像数据
            
            # 收集情绪数据
            emotion_data.append({
                'timestamp': current_time,
                'emotion': emotion,
                'probability': prob
            })
            
            # 检测高强度情绪（概率大于95%）并记录
            if prob > 95:
                # 格式化时间
                current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
                notification_msg = f"{current_time_str} 检测到高强度情绪：{emotion} ({prob:.0f}%)"
                
                # 将通知记录到日志文件
                log_file = 'emotion_log.txt'
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(notification_msg + '\n')
                
                print(f"已记录高强度情绪：{notification_msg}")

            # 显示情绪文本
            text = f'{emotion}: {prob:.0f}%'
            # 添加黑色边框增加对比度
            cv2.putText(frame, text, 
                       (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            # 在边框上方添加白色文本
            cv2.putText(frame, text, 
                       (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示视频窗口
        cv2.imshow('情绪检测', frame)
        
        # 按'q'键或ESC键退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27是ESC键的ASCII码
            break
    
    # 分析收集到的情绪数据
    if emotion_data:
        # 读取日志文件中的所有记录
        log_file = 'emotion_log.txt'
        with open(log_file, 'r', encoding='utf-8') as f:
            emotion_logs = f.readlines()
        
        # 调用Deepseek API进行分析
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if api_key and api_key != 'YOUR_API_KEY_HERE':
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                
                # 准备分析提示
                analysis_prompt = "分析以下情绪检测数据，找出主要情绪趋势和重要变化点：\n" + ''.join(emotion_logs)
                
                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "user", "content": analysis_prompt}
                    ]
                }
                
                response = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    analysis_result = response.json()
                    analysis_content = analysis_result['choices'][0]['message']['content']
                    
                    # 将分析结果写入日志文件
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"Deepseek分析：{analysis_content}\n\n")
                        
                    print("\n情绪数据分析结果：")
                    print(analysis_content)
                else:
                    print(f"API请求失败：{response.status_code}")
                    print(response.text)
                    
            except Exception as e:
                print(f"发送情绪分析请求失败：{e}")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()