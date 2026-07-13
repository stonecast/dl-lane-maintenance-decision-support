---
# 딥러닝 기반 차선 상태 분석 및 유지보수 의사결정 지원

기존 인력 중심 도로 점검 방식의 한계를 보완하기 위해
딥러닝 기반 영상처리를 활용하여 차선 상태를 분석하고
유지보수 우선순위를 제안하는 시스템을 구현한 학부 프로젝트입니다.

## 담당 역할

- Python 기반 딥러닝 모델 구현 및 학습
- 데이터 전처리 및 라벨링 기준 수립
- Django 기반 웹 시스템 구현
- TMAP API 연동
- 유지보수 우선순위 로직 구현
- 모델 성능 평가(IoU)

---

## 사용 기술

Python

PyTorch

OpenCV

Django

SQLite

TMAP API

---

## 전체 시스템 흐름 

도로 이미지 업로드부터 딥러닝 추론, 지도 기반 결과 시각화, 유지보수 우선순위 보고서 생성까지 하나의 워크플로우로 구성하였습니다.
(순서도 추가)

**1. 메인화면**  
시스템의 전체 현황을 확인하는 메인 화면입니다. 최근 업로드된 배치, 추론 진행 상태, 위치 정보를 한눈에 확인할 수 있습니다.
<img width="550" height="300" alt="image" src="https://github.com/user-attachments/assets/1873c9e1-52bf-44e7-896f-f5e9c4bb94b2" />

<br>

**2. 도로 이미지 업로드 및 배치 생성**  
도로 이미지를 배치 단위로 업로드하고 관리합니다. 위치 정보(CSV)를 함께 등록하면 추론 결과를 지도와 연동하여 확인할 수 있습니다.
<img width="450" height="250" alt="image" src="https://github.com/user-attachments/assets/cfd86fe3-1c54-4a39-855a-85ad6cea5c6d" />

<br>

**3. 추론 진행 및 추론 결과, 위치 시각화**  
배치 단위로 딥러닝 추론을 수행하고, 이미지별 정상/불량 판정 결과와 위치 정보를 지도 기반으로 시각화하여 제공합니다.
<img width="550" height="250" alt="image" src="https://github.com/user-attachments/assets/0c7e71f7-dd51-4240-a28c-5142d41364ea" />

<br>

**4. 유지보수 우선순위 보고서 생성**  
추론 결과와 실시간 교통 정보를 종합하여 구간별 차선 상태를 분석하고, 유지보수 우선순위를 보고서 형태로 제공합니다.
<br>
<img width="550" height="350" alt="image" src="https://github.com/user-attachments/assets/b371e427-e04c-4a7f-9758-eb2ca227faef" />

<br>

---
## 모델 구성

본 프로젝트는 차선 검출과 차선 결함 검출을 각각 독립적으로 수행하는
2단계 Segmentation 구조를 적용하였습니다.

단일 모델에서 정상/불량을 직접 분류하는 대신,
차선 영역과 결함 영역을 각각 분리하여 검출한 뒤
결함 비율을 계산하여 최종 정상/불량을 판단하도록 설계하였습니다.

또한 실제 서비스 형태로 구현하는 것을 목표로 하였기 때문에,
비교적 가볍고 구현이 용이한 U-Net을 기반 모델로 선정하였습니다.
<br>
        <img width="300" height="500" alt="image" src="https://github.com/user-attachments/assets/eaa70172-c7dc-4811-8175-f71eb945495a" />


1. Model_1  
    첫 번째 모델은 도로 이미지에서 차선 영역을 검출합니다.
  
    약 16,000장의 차선 Polygon 데이터로 학습하였으며, 
    차선은 형태적 특징이 비교적 명확하여 안정적인 성능을 확보할 수 있었습니다.
  
    IoU (Lane) : 92.8%
   
    <br>
    <img width="600" height="350" alt="image" src="https://github.com/user-attachments/assets/952fc34e-e958-4c5a-949b-99c74f5daed0" />

<br>

2. Model_2  
    두 번째 모델은 차선 내부의 결함 영역을 Segmentation합니다.
  
    결함 데이터는 직접 라벨링이 필요한 데이터였으며,
    학습 데이터의 수가 제한적이어서 Lane Segmentation에 비해 제한적인 성능을 보였습니다.
  
    <br>
    <img width="600" height="350" alt="image" src="https://github.com/user-attachments/assets/5f93dd8b-248b-4ca1-a5f3-18c855459c6a" />
  
<br>

두 모델의 결과를 결합하여 차선별 결함 비율을 계산하고, 
사용자가 설정한 임계값을 초과하는 경우 불량 차선으로 판단하였습니다.

1. 차선 세그멘테이션 결과를 통해 차선 영역을 추출
2. 결함 세그멘테이션 결과 중 차선 영역에 해당하는 픽셀만 선택
3. 하나의 차선 영역 내에서 결함 픽셀 비율을 계산
4. 결함 비율이 사용자가 설정한 임계값을 초과할 경우   해당 차선을 불량 차선으로 판단
<br><br>
<img width="600" height="350" alt="image" src="https://github.com/user-attachments/assets/130c06d7-cbc9-4449-a420-0db90391f3b4" />

<br>
- IoU (정상 차선): 0.8767<br>
- IoU (불량 차선): 0.8273<br>
- mIoU(정상/불량 기준): 0.8584%<br>
<br>

최종적으로 이미지 단위가 아닌
차선 단위의 정상/불량 결과를 제공하도록 구현하였습니다.

일반화 성능 확인을 위해 약 1,000장의 별도 이미지에 대해
차선을 정상/불량으로 재분류한 후,
해당 결과를 기반으로 정량 평가를 수행하였습니다.

---

### 프로젝트에서 고민했던 점

초기에는 하나의 모델에서 정상/불량을 직접 분류하는 방법도 고려하였습니다.

그러나 실제 유지보수에서는 차선 전체가 아닌
차선 내부 결함의 비율을 판단하는 것이 중요하다고 생각하여,

차선 검출과 결함 검출을 분리한
2단계 Segmentation 구조를 적용하였습니다.
<br>
### 프로젝트를 통해 배운 점

본 프로젝트를 진행하며 딥러닝 모델 자체의 성능뿐 아니라, 문제를 어떻게 정의하고 모델을 어떻게 활용하느냐가 결과를 크게 좌우한다는 점을 배웠습니다.

단순히 정상/불량을 직접 분류하는 방식 대신, 차선 검출과 결함 검출을 분리한 뒤 결함 비율을 계산하는 구조를 적용하면서 동일한 모델이라도 활용 방식에 따라 시스템의 성능과 활용도가 크게 달라질 수 있음을 경험했습니다.

또한 딥러닝 모델을 구현하는 것에 그치지 않고, Django 기반 웹 서비스와 외부 API를 연동하여 하나의 시스템 형태로 구현하는 과정을 경험할 수 있었습니다.

--- 

### 실행방법

#### for runserver
``` 
conda activate torch-gpu

cd [your dir]/lane_detector

python manage.py runserver
```
#### for create account

```
python manage.py createsuperuser
python manage.py changepassword <username>

default admin's id == admin / dldnjstjr123
```
#### 외부 API 키 설정
지도 및 교통 정보 시각화를 위해 외부 API 키가 필요합니다.
보안상의 이유로 API 키는 저장소에 포함되어 있지 않으며,
실행 환경에서 별도로 설정해야 합니다.<br>
경로: config/settings.py [line 14]
```
TMAP_APP_KEY = "TMAP API KEY"
```
