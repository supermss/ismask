# 사전설치 라이브러리 
# 1) pip install opencv-python
# 2) pip install cvlib
# Description : capture 함수를 통해 데이터셋 직접 생성
# 실행 : capture 폴더 안에 arguments 로 지정된 폴더에 저장
#        python caputre.py nomask
#        python caputre.py mask  

import cv2
import cvlib as cv
import time
import os
import sys

def capture(path, m=1):  #caputre 함수 생성 파라미터 m 미지정시 1개만 생성
    count = 0
    
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened(): #카메라가 없을때
        raise Exception("카메라 없음")
    
    while count < m:
      ret, frame = webcam.read() #2개의 리턴값을 튜플로 반환함.
      if not ret:
          raise Exception("캡쳐가 없음")
                  
      faces, confidences = cv.detect_face(frame) #얼굴찾기
        
      for face, conf in zip(faces, confidences):
        if conf < 0.8:
            continue
        start_x, start_y, end_x, end_y = faces[0]
        cv2.imwrite(path+str(count)+'.jpg', frame[start_y:end_y, start_x:end_x, :]) #이미지 저장
        count += 1
      
      time.sleep(0.3) # 캡쳐간 시간 0.3로 지연 
    

    print(count, end='') #캡쳐 완료 시 
    webcam.release()

if __name__ == '__main__':
    arguments = sys.argv

    if len(arguments) == 1:
        print("No Argument")
    else :
      input_path = arguments[1]

      dir_name = "capture"
      if not os.path.exists(dir_name+"/"+input_path):
          os.mkdir(dir_name+"/"+input_path)

      capture(dir_name+"/"+input_path+"/",300)