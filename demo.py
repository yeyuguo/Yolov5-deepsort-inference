from AIDetector_pytorch import Detector
import imutils
import cv2
import os
import subprocess
import shutil

def main():
    func_status = {}
    func_status['headpose'] = None
    
    name = 'demo'

    det = Detector()
    cap = cv2.VideoCapture('/Users/bytedance/Code/github/Yolov5-deepsort-inference/test01.mp4')
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("错误：无法打开视频文件")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print('fps1:', fps)
    if fps <= 0:
        fps = 30  # 使用默认帧率
    print('fps2:', fps)
    t = int(1000/fps)

    # 创建临时文件夹存储帧
    temp_folder = 'temp_frames'
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.makedirs(temp_folder)
    
    # 初始化帧计数器
    frame_count = 0
    
    print("开始处理视频...")
    
    while True:
        ret, im = cap.read()
        # print('ret, im:', ret, im)
        if not ret or im is None:
            print(f"视频处理完成，共处理 {frame_count} 帧")
            break
        
        # 增加帧计数
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"已处理 {frame_count} 帧")
        
        # 处理当前帧
        try:
            result = det.feedCap(im, func_status)
            result = result['frame']
            result = imutils.resize(result, height=500)
            
            # 保存当前帧为图像文件
            frame_path = os.path.join(temp_folder, f'frame_{frame_count:06d}.jpg')
            cv2.imwrite(frame_path, result)
            
            # 显示结果
            cv2.imshow(name, result)
            key = cv2.waitKey(t)
            
            # 按 ESC 或 'q' 键退出
            # if key == 27 or key == ord('q') or cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            #     print("用户中断处理")
            #     break
                
        except Exception as e:
            print(f"处理第 {frame_count} 帧时出错: {e}")
            # 继续处理下一帧，而不是退出循环
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print('frame_count:', frame_count)
    # 使用 FFmpeg 将图像序列转换为视频
    if frame_count > 0:
        print("正在使用 FFmpeg 生成视频...")
        try:
            # 检查 FFmpeg 是否可用
            subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 使用 FFmpeg 生成视频
            subprocess.run([
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', os.path.join(temp_folder, 'frame_%06d.jpg'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',  # 控制质量，值越小质量越高
                'result.mp4'
            ], check=True)
            
            print("视频生成成功：result.mp4")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg 错误: {e}")
            print("尝试使用 OpenCV 生成视频...")
            
            # 如果 FFmpeg 不可用，尝试使用 OpenCV
            try:
                first_frame = cv2.imread(os.path.join(temp_folder, 'frame_000001.jpg'))
                h, w = first_frame.shape[:2]
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('result.mp4', fourcc, fps, (w, h))
                
                for i in range(1, frame_count + 1):
                    frame_path = os.path.join(temp_folder, f'frame_{i:06d}.jpg')
                    if os.path.exists(frame_path):
                        frame = cv2.imread(frame_path)
                        out.write(frame)
                
                out.release()
                print("使用 OpenCV 生成视频成功")
            except Exception as e:
                print(f"使用 OpenCV 生成视频失败: {e}")
        except FileNotFoundError:
            print("FFmpeg 未安装，尝试使用 OpenCV 生成视频...")
            # 同上，使用 OpenCV 生成视频的代码
        
        # 清理临时文件
        print("清理临时文件...")
        # 如果需要保留临时文件用于调试，可以注释掉下面这行
        # shutil.rmtree(temp_folder)
    
    print("处理完成")

if __name__ == '__main__':
    main()