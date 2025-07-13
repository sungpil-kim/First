# 텐서플로어 기초 강의 요약

## 1. 텐서플로어 기초
import tensorflow as tf

# 1차원 구조
텐서 = tf.constant([3,4,5])
텐서2 = tf.constant([6,7,8])
텐서3 = tf.zeros([3,2,3])

# 2차원 구조(행렬)
텐서4 = tf.constant([[1,2,3],
                 [4,5,6]])    

# 출력 예시
print(텐서+텐서2)
# 출력값 : [9,11,13]

print(텐서3)
# 출력값 : #[[[0,0,0]]] [[[0,0,0]]] [[0,0,0]]]

print(텐서4.shape)
# 출력값 : #(2,3)

## 2. 텐서플로어 실습(1) - 경사하강 기법을 활용한 반복학습으로 신발 사이즈 구하기 실습
import tensorflow as tf

# 입력 및 실제값 설정 (float32로 설정)
키_학습 = tf.constant(170.0, dtype=tf.float32)
신발_학습 = tf.constant(260.0, dtype=tf.float32)

# 학습할 변수 (a: 기울기, b: 절편)
a = tf.Variable(0.1, dtype=tf.float32)
b = tf.Variable(0.2, dtype=tf.float32)

# Adam 옵티마이저 설정
opt = tf.keras.optimizers.Adam(learning_rate=0.01)

# 모델 학습 단계 함수
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        # 예측값 = 키 * a + b
        예측값 = 키_학습 * a + b
        # 손실 계산 (평균 제곱 오차)
        loss = tf.square(신발_학습 - 예측값)

    # 기울기 계산 및 적용
    gradients = tape.gradient(loss, [a, b])
    opt.apply_gradients(zip(gradients, [a, b]))
    
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
    
    print(f"\n입력하신 키 {사용자_키:.1f}cm에 대한 예상 신발 사이즈: {예상_신발_사이즈:.1f}mm")

except ValueError:
    print("잘못된 입력입니다. 숫자를 입력해 주세요.")
