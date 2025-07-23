# 배관 및 기기 공종의 중랍 형식의 플랜트 3D CAD 모델 가시화 시스템 개발
 - 저자 : 김세윤, 이현오, 김병철, 문두환
 - 학회 : 한국CDE학회 논문집 (Korean Jounal of Computational Design and Engineering)

 
## 1. Introduction
 - 공정 플랜트의 상세설계를 위한 플랜트 3D CAD 모델의 공유 방법으로 중립 모델을 사용하는 방안이 있다. 중립 형식의 플랜트 모델은 상세 설계 정보 확인에 이점이 있지만 기능의 제한 문제점이 있다. 따라서 3D CAD 모델의 가시화를 통해서 여러 설계 정보를 확인하는 시스템 개발이 필요하다.
 ⇒ 이는 실제 다른 산업에서도 동일한 경우라고 생각한다. 자동차 산업의 경우에도 차량의 전체적인 데이터 취합을 위해서 .CATpart가 아닌 .cgr로 데이터를 교환한다. 이는 간단하게 각 부품의 위치와 간섭여부를 확인할때는 용이하지만 간섭량 파악등의 여러 기능을 사용할때 제한된다.
 
## 2. Netral Plant 3D CAD Models
 - 중립 3D 설계 정보 모델의 구조는 Plant System이라는 틀 안에 배관 공종인 Piping System과 기기 공종의 Mechanical System으로 설계정보가 구분된다. Piping System에는 각각의 Piping들의 집합으로 Segment의 연결 및 참조, 설계 속성 정보를 포함하고 있다. 이와 유사하게 Mechanial System의 경우 기기의 정보들이 포함되어 있다. 이때 중립 3D 설계 정보의 입력파일은 비형상 정보(XML 형식)과 형상 정보(SAT)로 구분된다. 비형상 정보에는 식별, 스펙, 형상, 연결 정보에 관한 참조 정보를 포함하고 형상 정보의 경우 branch단위로 구분되는 형상 정보를 포함하고 있다.
 
 - 저자는 가시화를 위해 중립 메타 모델을 확장한 스펙-카탈로그 모델로 변환하였다. 기존의 중립 모델이 Piping system이라는 큰 틀에서 분류되었다면, 해당 모델을 통해서 각각의 기자재를 식별하였다. 중립 스펙-카탈로그 모델은 스펙, 카탈로그, 분류체계 모델로 구성된다. 각각의 모델에 대한 설명은 아래와 같다.

스펙 모델(Spec model)
스펙(SPEC) : 조건별로 해당되는 카탈로그의 식별 역할
필터(SelectionFilter) : 기자재 선정을 위한 값 저장
필터클래스(SelectionFilterClass) : 기자재를 타입별로 선정하기 위한 속성 저장
속성 값(AttributeValue) : 속성에 대한 값을 저장하는 정보 객체
속성(Attribut) : 기자재 선정 시 필요한 속성 정보 저장 객체

중립 분류체계 모델(Part classification model)
\n클래스(Class) : 기자재 타입에 따른 사양 정보 저장
사양(Property) : 기자재 타입에 따른 사양 정보 저장
코드마스터(CodeMaster) : 기자재에 대한 속성 및 사양 정보에서 곹오적으로 기자재에서 사용되는 값들을 Code의 형태로 저장하는 정보 객체
코드(Code) : 값 단위(UnitOfValue), 열거 항목(EnumerationItem), 열거 유형(EnumerationType)

 중립 카탈로그 모델(Catalog Model)
  카탈로그(Catalog) : 특정 타입에 해당되는 기자재의 사양 정보 앖의 목록, 형상 정보 등이 저장
  사양 정보 값(Property Value) : 사양 정보에 대한 값이 저장

  <img width="1923" height="927" alt="Image" src="https://github.com/user-attachments/assets/fe816f07-abb9-40ea-a625-f9131857abd0" />
