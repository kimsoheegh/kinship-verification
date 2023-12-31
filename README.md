# 가족관계 예측 모델
![그림3](https://github.com/kimsoheegh/kinship-verification/assets/91236577/41208082-c9ba-43a3-838f-d34997ae3a84)
<br><br>
### :mag: 목적
얼굴 인식 모델은 대부분 서양인의 얼굴 데이터를 기반으로 하고 있다. 때문에 동양인의 얼굴을 통한 안면 인식의 경우 정확도가 다소 떨어질 수 있다. 따라서 모델의 범용적 사용을 위해 부족한 동양인 데이터 세트를 새로 구축하고 이를 활용하여 높은 안면 인식 정확도를 달성한다. 여러 모델이 나와있지만 이를 활용한 응용 서비스 생각보다 많지 않다. 동양인 데이터 기반의 안면 인식 모델을 통해 나이 예측 모델과 가족 예측 모델을 개발한다면 이를 활용한 서비스들이 가족 관계 관련 산업 분야에서 활용될 것이라 예상한다.
<br><br><br>
## 가족관계 예측 네트워크 학습
* 기본 Siamese 네트워크 코드 : [링크](https://github.com/kimsoheegh/kinship-verification/blob/master/kinship_verification/kinship_verification.py)
* 가중치 기반의 Siamese 네트워크 코드(테스트 정확도 60.24%) : [링크](https://github.com/kimsoheegh/kinship-verification/blob/master/kinship_verification/weighted_kinship_verification.py)
<br>

### :mag: 결과물
![image](https://github.com/kimsoheegh/kinship-verification/assets/91236577/7235a8b1-71ad-4839-b905-10c49a3936d3)
