import cv2
import zmq
import numpy as np
import mediapipe as mp
import json
import time
import sys

# 配置
Param_InPort  = "6000" # 接收 Camera (SUB)
Param_OutImg  = "6001" # 发送 Preview (PUB)
Param_OutPose = "6002" # 发送 Keypoints (PUB)
Param_IP      = "127.0.0.1"

def main():
    print(f"[Py] Starting Inference Engine...")
    
    # 1. Setup ZMQ
    context = zmq.Context()
    
    # Receiver: Camera Frames
    socket_sub = context.socket(zmq.SUB)
    socket_sub.connect(f"tcp://{Param_IP}:{Param_InPort}")
    socket_sub.setsockopt_string(zmq.SUBSCRIBE, "") # 订阅所有
    
    # Publisher: Preview Image
    socket_pub_img = context.socket(zmq.PUB)
    socket_pub_img.bind(f"tcp://*:{Param_OutImg}")
    
    # Publisher: Keypoints
    socket_pub_pose = context.socket(zmq.PUB)
    socket_pub_pose.bind(f"tcp://*:{Param_OutPose}")

    # 2. Setup Mediapipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    print("[Py] Ready and waiting for frames...")

    while True:
        try:
            # 3. Receive Frame (Multipart: Header + JPEG Bytes)
            # 使用 NOBLOCK 避免死锁，实际使用中可以用 Poller
            if socket_sub.poll(10): 
                msg = socket_sub.recv_multipart()
                # msg[0] 是 metadata (JSON), msg[1] 是图片数据
                
                # Decode Image
                np_arr = np.frombuffer(msg[1], np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue

                # 4. Inference
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # 5. Prepare Keypoints Data
                kp_list = []
                if results.pose_landmarks:
                    # Draw Skeleton on frame
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    # Extract 3D Keypoints (x, y, z, visibility)
                    for lm in results.pose_world_landmarks.landmark:
                        kp_list.extend([lm.x, lm.y, lm.z, lm.visibility])

                # 6. Send Preview Image (JPG)
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                
                meta_img = json.dumps({"w": frame.shape[1], "h": frame.shape[0], "ts": time.time()})
                socket_pub_img.send_multipart([meta_img.encode('utf-8'), buffer.tobytes()])

                # 7. Send Keypoints (Binary Float32)
                if kp_list:
                    kp_array = np.array(kp_list, dtype=np.float32)
                    meta_pose = json.dumps({"count": len(results.pose_world_landmarks.landmark)})
                    socket_pub_pose.send_multipart([meta_pose.encode('utf-8'), kp_array.tobytes()])
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[Py Error] {e}")
            break

    print("[Py] Shutting down.")
    socket_sub.close()
    socket_pub_img.close()
    socket_pub_pose.close()
    context.term()

if __name__ == "__main__":
    main()