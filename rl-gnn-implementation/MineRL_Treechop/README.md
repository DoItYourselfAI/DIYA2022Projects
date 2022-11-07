# Setup

1. Virtualenv setup with python 3.8
2. Install python dependencies (`pip install -r requirements.txt`)
3. Install JAVA 1.8
    - mac의 경우 "1.8.0_222"에서 실행 성공
    - (caution) JAVA 최신버젼을 사용하는 경우 오히려 깨지는 경우가 존재함


# TODO List

- implement make_action function
- environment wrapping (frameskip, framestack) 
  
- [X] 1. MDP 및 환경 정보 확인
- [X] 2. Random Policy 구현 및 동작 확인
- [X] 3. 모델 구현
- [X] 4. 모델 동작 확인
- [X] 5. 버퍼 구현 (Replay Buffer, Temporal Buffer 등)
- [X] 버퍼 저장 동작 구현
- [ ] 6. 모델 업데이트 코드 구현
- [ ] 7. 하이퍼파라미터 조정
- [ ] 8. 반복 실험  
