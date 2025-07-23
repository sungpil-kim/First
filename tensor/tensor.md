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
## 3. 텐서플로어 외부 데이터 가져오기 기초

### (1) 빈칸 삭제 여부 확인 코드
```
import pandas as pd

# 1. 파일 읽기
data = pd.read_csv('gpascore.csv')

# 2. 빈칸이 있는지 확인
print(data.isnull().sum())
# 출력값 - gre 1 => gre 항목에 1개의 빈칸

# 3. 빈칸을 삭제
data = data.dropna()

# 4. 빈칸 삭제 여부 확인
print(data.isnull().sum())
```
### (2) 빈칸에 원하는 값을 채우고 확인 코드
```
import pandas as pd

# 1. 파일 읽기
data = pd.read_csv('gpascore.csv')

# 2. 빈칸이 있는지 확인
print(data.isnull().sum())

# 3. 빈칸이 있는 행의 인덱스를 식별
# axis=1 : 행(row) 방향 적용
# axis=0 : 열(column) 방향 적용
# data[data.isnull().any(axis=1)].index => 빈칸이 있는 행의 인덱스 번호를 반환
# data.isnull().any(axis=1) => 빈칸이 있는 행의 T/F 반환
null_indices = data[data.isnull().any(axis=1)].index

# 4. 빈칸을 100으로 채우고 data_filled 변수에 저장
data_filled = data.fillna(100)

# 5. 빈칸이 있던 행만 출력하여 100으로 채워졌는지 확인
# .loc : 해당 인덱스에 해당하는 모든 행 선택
print(data_filled.loc[null_indices])
```

### (3) 원하는 열만 출력 방법
```
import pandas as pd

# 1. 파일 읽기
data = pd.read_csv('gpascore.csv')

# 2. 원하는 열의 이름 입력
print(data['gpa'])

# 3. 원하는 열의 최솟값 파악
print(data['gpa'].min())

# 4. 원하는 열의 개수 파악
print(data['gpa'].count())
```

## 4. 텐서플로어 실습(2) - csv 데이터를 활용하여 대학원 합격 예측 실습
```
import pandas as pd
import numpy as np
import tensorflow as tf
# 데이터 표준화 수행 도구
# 입력 데이터를 변환하여 평균이 0, 표준편차가 1
from sklearn.preprocessing import StandardScaler

# 1. 파일 읽기
# 'gpascore.csv' 파일이 필요합니다.
try:
    data = pd.read_csv('gpascore.csv')
except FileNotFoundError:
    print("Error: 'gpascore.csv' 파일을 찾을 수 없습니다. 파일을 업로드하거나 경로를 확인해주세요.")
    exit()

# 2. 빈칸 삭제
data = data.dropna()

# 3. 데이터 분리 및 전처리 개선
# 목표 변수 (Target Variable) - 해당 변수를 통해 모델 예측 정확한지 평가 및 개선
y데이터 = data['admit'].values

# 특징 변수 (Features) - gre, gpa, rank를 토대로 목표 변수 예측
# pandas의 열 선택을 사용, x데이터를 추출
x데이터_df = data[['gre', 'gpa', 'rank']]

# 4. 데이터 표준화 (Standardization)
# StandardScaler를 사용하여 특징 데이터의 스케일을 조정
scaler = StandardScaler()
x데이터_scaled = scaler.fit_transform(x데이터_df)

# 텐서플로우 모델에 입력하기 위해 넘파이 배열로 변환
x데이터 = np.array(x데이터_scaled)

# 5. 텐서플로우 모델 정의 및 훈련
# 활성화 함수를 'relu'로 변경하여 성능을 개선할 수 있습니다.
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# binary_crossent : 이진 분류 문제에 사용되는 표준 손실 함수
# metrics=['accuracy'] : 평가 지표 - 올바르게 예측한 비율
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("모델 훈련 시작...")
model.fit(x데이터, np.array(y데이터), epochs=1000)
print("모델 훈련 완료.")

# 6. # 사용자 입력 받아서 성적에 따른 합격 여부 예측
try:
    # 사용자로부터 쉼표로 구분된 GRE, GPA, RANK 값 입력받기
    user_input_str = input("\n[GRE, GPA, RANK] 순서로 쉼표(,)를 사용하여 입력하세요 (예: 700, 3.5, 2): ")
    
    # 입력된 문자열을 쉼표로 분리하고 각각 숫자로 변환
    # 이 과정에서 값이 3개가 아니거나 숫자가 아니면 ValueError가 발생
    input_values = [float(x.strip()) for x in user_input_str.split(',')]
    
    # 입력된 값들을 넘파이 2차원 배열 형태로 변환
    user_data_np = np.array([input_values])

    # 훈련 시 사용한 scaler로 예측 데이터를 표준화
    # (이전 단계에서 scaler.fit_transform()을 통해 정의된 scaler 사용)
    user_data_scaled = scaler.transform(user_data_np)

    # 예측 수행
    prediction = model.predict(user_data_scaled)
    
    # 예측 결과 출력
    print("\n예측 결과 (합격 확률):", prediction[0][0])
    
    # 0.5를 기준으로 합격/불합격 여부 판단
    if prediction[0][0] > 0.5:
        print("예측: 합격할 것으로 예상됩니다.")
    else:
        print("예측: 불합격할 것으로 예상됩니다.")

# 입력 형식이 잘못되었을 때 발생하는 오류 처리
except ValueError:
    print("\n입력 오류: 유효한 숫자 3개를 쉼표(,)로 구분하여 입력해주세요.")
except Exception as e:
    print(f"\n예상치 못한 오류가 발생했습니다: {e}")
```
