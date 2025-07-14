# 텐서플로어 기초 강의 요약

## 1. 텐서플로어 기초
```python
import tensorflow as tf

# 1차원 구조
텐서 = tf.constant([3,4,5]) # constant는 '상수'로 불변하는 값
텐서2 = tf.constant([6,7,8])
텐서3 = tf.zeros([3,2,3])

# 2차원 구조(행렬)
텐서4 = tf.constant([[1,2,3],
                 [4,5,6]])    

# 출력 예시
print(텐서+텐서2)
# 출력값 : [9,11,13]

print(텐서3)
# 출력값 : 
[[[0. 0. 0.]
  [0. 0. 0.]]

 [[0. 0. 0.]
  [0. 0. 0.]]

 [[0. 0. 0.]
  [0. 0. 0.]]]

print(텐서4.shape)
# 출력값 : (2,3)
```

## 2. 텐서플로어 실습(1) - 경사하강 기법을 활용한 반복학습으로 신발 사이즈 구하기 실습
```python
import tensorflow as tf

# 입력 및 실제값 설정 (float32로 설정 - 정확도 및 효율성)
키_학습 = tf.constant(170.0, dtype=tf.float32)
신발_학습 = tf.constant(260.0, dtype=tf.float32)

# 학습할 변수 (a: 기울기, b: 절편)
a = tf.Variable(0.1, dtype=tf.float32)  # Variable은 '변수'로 학습을 통해 업데이트
b = tf.Variable(0.2, dtype=tf.float32)

# Adam 옵티마이저 설정
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
#Adam (Adaptive Moment Estimation): 개별 매개변수에 맞춰 학습률을 조절하여 빠르고 안정적인 학습을 돕는 강력한 최적화 기법.
#SGD (확률적 경사 하강법, Stochastic Gradient Descent): 가장 기본적인 최적화 기법
#RMSprop (Root Mean Square Propagation): Adagrad의 변형으로, 기울기의 크기를 조절하여 학습을 안정화.
#Adagrad (Adaptive Gradient Algorithm): 자주 발생하는 매개변수에는 작은 학습률을, 드물게 발생하는 매개변수에는 큰 학습률을 적용


# 모델 학습 단계 함수
@tf.function
#그래프 모드 (Graph Mode): 연산 과정을 하나의 실행 가능한 그래프로 컴파일하여 실행
def train_step():
    with tf.GradientTape() as tape:
        # 예측값 = 키 * a + b
        예측값 = 키_학습 * a + b
        # 손실 계산 (평균 제곱 오차)
        loss = tf.square(신발_학습 - 예측값)

    # 기울기 계산 및 적용
    gradients = tape.gradient(loss, [a, b])
    # 변수 a와 b에 대한 loss의 기울기를 구함
    opt.apply_gradients(zip(gradients, [a, b]))
    # 계산된 기울기(gradients)를 모델의 변수(a,b)에 적용하여 업데이트
    
    return loss

## 학습 실행 (모델 최적화)
print("모델 학습 시작...")
for i in range(1000):
    loss = train_step()
    if i % 200 == 0:
        # 학습 진행 상황 출력
        print(f"Epoch {i}: Loss = {loss.numpy():.4f}, a = {a.numpy():.4f}, b = {b.numpy():.4f}")

print("\n모델 학습 완료.")
print(f"최종 a (기울기): {a.numpy():.4f}")
print(f"최종 b (절편): {b.numpy():.4f}")

## 예측 함수 추가

# 학습된 a와 b를 사용하여 신발 사이즈 예측 함수 정의
def 예측_신발_사이즈(입력_키):
    # 예측 신발 사이즈 = 입력_키 * 학습된 a + 학습된 b
    예측값 = 입력_키 * a.numpy() + b.numpy()
    return 예측값

# 사용자 입력 받아서 신발 사이즈 예측
try:
    사용자_키 = float(input("\n예측할 키를 입력하세요 (cm): "))
    예상_신발_사이즈 = 예측_신발_사이즈(사용자_키)
    # 사용자_키의 값을 입력_키로 전달 (사용자_키 = 입력_키)
    # 입력_키는 매개변수, 이를 통해서 가독성 향상 및 사용자_키 재사용 등의 다양항 상황에 적용 가능
    
    print(f"\n입력하신 키 {사용자_키:.1f}cm에 대한 예상 신발 사이즈: {예상_신발_사이즈:.1f}mm")

except ValueError:
    print("잘못된 입력입니다. 숫자를 입력해 주세요.")
    ```
