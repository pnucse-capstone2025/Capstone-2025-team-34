### 1. 프로젝트 배경
AI의 사용이 일반화되어가는 현세대에서, 이제 AI 및 LLM을 다루는 능력은 현대인들에게 필수적인 능력이 되어가고 있다. 이제는 LLM을 원하는 분야로 조정하기 위해 데이터셋을 제작하고, 이를 학습시키며, 특정 분야에 특화된 서비스를 만드는 역량은 개발자들에게 있어 필수가 될 것이다.

Re:Fresh 팀은 LLM을 활용하여 영어 객관식 문제를 풀어 답을 생성하는 서비스의 개발을 주제로 하여 졸업과제를 진행하였다.

#### 1.1. 국내외 시장 현황 및 문제점
국내외적으로 LLM을 활용한 서비스를 만드는 것은 현재 매우 활성화 되어 있다.
많은 분야에 AI 모델을 적용할 수 있지만 아직 대중적으로 널리 활성화되지 않은 분야들이 많다.
영어 객관식 문제 풀이도 그 중 하나로, 해당 분야에 특화된 서비스는 미비한 실정이다.

#### 1.2. 필요성과 기대효과
해당 프로젝트는 일상생활에서 마주칠 수 있는 문제 중 하나인 "영어 객관식 문제 풀이"에 특화된 모델을 제시함으로서, 향후 비슷한 서비스에 대해 개발 방향성을 제시할 수 있을 것으로 기대된다.

---

### 2. 개발 목표
#### 2.1. 목표 및 세부 내용
- LLM을 활용하여 영어 객관식 문제를 풀어 답을 생성하는 서비스의 시연
- LLM이 생성한 답의 정확도를 높일 수 있는 방법을 연구하고 적용

#### 2.2. 기존 서비스 대비 차별성 
여러 모델들을 활용하여 특정 모델에 과편향되지 않고 답을 도출하기에, 생성된 정답에 대한 평균적인 신뢰도를 높이려고 하였다. 

#### 2.3. 사회적 가치 도입 계획 
프로젝트는 모델들에 대해 각각 문제풀이를 하게 하고 그 결과들을 종합하는 방식으로 이루어져, 새로운 모델을 추가하는 것에 대해 확장 가능성이 열려 있다. 또한 모델 개별이 문제에 대한 답을 생성하도록 하였기 때문에, 원한다면 본인이 직접 사용 모델을 편집하여 발전된 서비스를 만들 수 있을 것이다.

---

### 3. 시스템 설계
#### 3.1. 시스템 구성도

<img width="1920" height="1080" alt="문제 전달" src="https://github.com/user-attachments/assets/82482f22-03fa-47b1-9a1b-2d5f49b3aead" />

- 사용자가 문제의 지문과 문제, 보기를 입력하여 서버에 전송
- 서버는 문제를 모델 서버에 전달하고, 모델 서버는 받은 문제에 대해 문제 분류 및 문제 풀이 실시
- 푼 결과를 종합하여 최종 답을 생성하고, 이를 다시 사용자에게 전달

#### 3.2. 사용 기술

##### 프론트엔드 : TypeScript, React, React Router
- React를 활용하여 웹 UI를 통해 사용자가 사용하기 쉬운 프론트엔드를 제공

##### 백엔드 : Java 21, Spring Boot, Gradle
- Java 및 Spring Boot를 활용하여 모델서버와 프론트를 연결하는 중계서버 구축

##### 모델 서버 : Python 3.10, FastAPI, Transformer
- Python과 Transformer를 활용하여 모델 학습
- FastAPI를 활용하여 웹 API 형태로 문제 수신 및 모델 생성 결과 송신

---

### 4. 개발 결과
#### 4.1. 전체 시스템 흐름도
- 프론트에서 문제 요청 → 서버를 통해 모델 서버로 전달 → 모델 서버에서 문제 분류 및 문제 풀이 진행 → 결과 및 분석 내용 전송

#### 4.2. 기능 설명 및 주요 기능 명세서

> 프론트엔드
- 영어 문제에 대한 정보 입력
- 지문, 문제, 보기를 입력하여 문제 전송 가능
- 전송 후 결과에 대해 풀이 모델이 어떤 과정을 거쳐 이러한 답을 도출하였는지 시각적으로 확인 가능

> 백엔드
- 프론트엔드에서 전송한 답을 받아 모델 서버로 전달 및 모델 서버에서 도출한 응답을 프론트엔드로 전달

> 모델 서버
- 백엔드 서버에서 전달된 문제에 대한 정보를 바탕으로 문제풀이를 실시

  1. 분류 모델
    - 문제의 지문과 문제, 보기에 대한 정보를 입력으로 받아, “이 문제가 어떤 종류의 문제에 해당하는가?”를 확률 값으로 출력
    - 해당 확률 값을 파싱하여 모델이 분류한 문제의 종류가 결정됨
  
  2. 풀이 모델 공통
    - 문제의 지문과 문제, 보기에 대한 정보를 입력으로 받아, “이 문제의 답이 어떤 것인가?”를 각 모델마다 확률 값으로 출력
  
  3. 최종 정답 생성
    - 각 모델별 생성한 확률 값과, 분류 모델이 분류한 문제의 종류에 따라 가중치를 결정
    - 해당 가중치를 반영하여 확률들을 결합하여 최종적으로 보기가 답일 확률에 대한 확률 값을 출력
    - 확률 값을 파싱하여 최종적으로 모델이 판단한 답을 확인 가능

#### 4.3. 디렉토리 구조
> 디렉토리 구조는 다음과 같이 되어 있다.
```
ㄴ front // 사용자 화면, 프론트엔드 내용 포함
ㄴ backend // 중계 서버, 백엔드 내용 포함
ㄴ model-server //실제 모델 서버, 본 프로젝트의 핵심 내용이자 문제 풀이 모델의 학습과 실행과 관련된 코드 포함
```

---

### 5. 설치 및 실행 방법
#### 5.1. 설치 절차 및 실행 방법

> 사전에 git 설치 및 리포지토리 클론 필요

```shell
git clone (리포지토리 주소)
```

---

> 프론트엔드

- 사전에 node.js(개발 환경 기준 `v22.17.1`) 및 pnpm(개발 환경 기준 `10.13.1`) 설치 필요
- 다음 명령으로 의존성 설치

```shell
pnpm install
```

서버 주소에 따라 코드 내의 내용 일부 수정 필요
front/app/routes/question.tsx에서,
```tsx
const SERVER_ENDPOINT = "http://(백엔드 서버 주소):8080/api/problem";
```

(백엔드 서버 주소)부분을 실제 백엔드의 ip로 수정해야 함

이후 front 폴더에서,
```shell
pnpm run build

pnpm start
```

순서대로 입력해 실행

---

> 백엔드

- Java 21 사전 설치 필요
- 사전에 ssh 터널링 필요

본 프로젝트에서는 9009번 포트로 터널링 하여 로컬에서도 접근 가능하도록 세팅

```shell
ssh -L 9009:localhost:8009 (사용자 계정)@(아이피)
(계정에 대한 암호 입력)
```

현재 계정은 학과 서버에서 발급 받은 계정을 사용한 것

intellij 설치되어 있는 환경일 경우 → GraduateServerApplication.java 바로 실행하면 서버 준비 ok

그렇지 않은 환경일 경우 → gradlew를 통해 jar파일 생성 후 해당 jar 파일 실행

```shell
./gradlew clean bootJar
java -jar build/libs/graduate-server-0.0.1-SNAPSHOT.jar
```

실행 후 서버 ip의 8080번 포트를 통해 웹 api 송수신 진행

---

> 모델 서버

- python 3.10.x 사전 설치 필요
- ipynb 파일 실행을 위한 Jupyter Notebook 설치 필요

1. .env 파일 생성

포함 내용
```python
HUGGINGFACE_TOKEN: {허깅 페이스 토큰}
GOOGLE_API_KEY: {구글 Gemini api 토큰}
```
2. 데이터 셋 구성

- 오픈데이터셋 정책 [9] [10] 에 따라 데이터셋의 재게시가 불가능하기 때문에 현재 GitHub에 업로드 되어 있지 않음
- 다만 make_dataset.ipynb 파일에 오픈데이터셋을 활용하고 작업하는 코드가 있으므로, 이를 참고하여 본인만의 데이터셋을 구축하여 사용하여도 됨
- 혹은 빠른 테스트를 위해 가짜 데이터셋을 만드는 코드를 사용할 수 있음
  - Capstone-2025-team-34/model_server/dataset/make_mock_dataset.ipynb 노트북 파일을 전부 실행

3. 개별 모델 학습 및 평가

각 model_server밑의 개별 모델 디렉토리

```
gemma-fintuning
gemma-STaR
gemma-teacher-student
phi-finetuning
phi-STaR
phi-teacher-student
```

에서 각각 실행

- 모델 학습을 위해 개별 모델 디렉토리의 train.py 파일을 실행
  - 사용하는 데이터셋의 크기에 따라 코드 내 수치 조정 필요
  - ***-fintuning
  - <img width="652" height="558" alt="image (2)" src="https://github.com/user-attachments/assets/8a2ad244-d356-4d63-847e-bac13eb4a9be" />
  - ***-STaR
  - <img width="1230" height="102" alt="image (3)" src="https://github.com/user-attachments/assets/31150257-13d9-4711-aed3-f8bc603a4a19" />
  - ***-teacher-student
  - <img width="1216" height="124" alt="image (4)" src="https://github.com/user-attachments/assets/59006cb2-d494-4c76-9f85-96c4c657989c" />

- 각 개별 모델의 테스트를 위해 개별 모델 디렉토리의 model_test.ipynb 노트북 파일을 실행
  - 사용하는 데이터셋의 크기에 따라 코드 내 수치 조정 필요
  - <img width="1642" height="362" alt="image (5)" src="https://github.com/user-attachments/assets/b291b5eb-e9a2-47ab-9e02-b612e0f67245" />

4. 분류 모델

Capstone-2025-team-34/model_server/classifier/ 하위의 학습 코드 2개를 각각 실행
만약 데이터셋의 위치가 수정되었다면, `base_paths`, `data_paths`를 각각 올바르게 수정

5. fast api 서버 구동
Capstone-2025-team-34/model-server/_final/fastapi 디렉토리로 이동한 후 아래 명령으로 파일 실행

```shell
python serving.py
```

이제 해당 서버의 8009번 포트를 통해 모델 서버의 결과값 송수신이 가능

---

#### 5.2. 오류 발생 시 해결 방법
- 패키지 미설치 오류 발생 시 - 요구 패키지 설치
- 모델 서버의 경우 온전한 실행을 위해 최소 VRAM 24GB 이상의 환경이 필요함

---

### 6. 소개 자료 및 시연 영상
#### 6.1. 프로젝트 소개 자료

[2025전기 졸업과제 발표영상_34_ReFresh_영어 객관식 문제 자동정답 생성기 개발.pptx](https://github.com/user-attachments/files/22607962/2025._34_ReFresh_.pptx)

#### 6.2. 시연 영상

[![Re:Fresh 발표 및 시연 영상](https://img.youtube.com/vi/zF91HYHA-iY/0.jpg)](https://www.youtube.com/watch?v=zF91HYHA-iY)

---

### 7. 팀 구성
#### 7.1. 팀원별 소개 및 역할 분담

|강형원|박규태|주연학|
|----|----|----|
|프로젝트 전체 총괄|문제 분류 모델 제작 및 적용|문제 풀이 모델 제작 및 적용|
|데이터셋 조사/제작/정제|프론트엔드 구축|모델 서버 구축|
|백엔드 서버 구축|||

#### 7.2. 팀원 별 참여 후기

|강형원|박규태|주연학|
|----|----|----|
|실제 데이터셋에 대해 조사하면서 적절한 데이터셋을 선정하는 것이 꽤나 어려웠으며, 선정 이후에도 데이터셋의 무결성을 검증하기 위해 데이터셋들을 일일이 검증하는 것이 꽤나 힘든 일이었다. 그러나, 실제로 이렇게 만든 데이터셋을 통해 모델에 파인튜닝하고 그 결과가 실제로 가시적으로 보이자 뿌듯함도 있었다.|BERT 계열의 모델을 이 과제를 통해 처음 접하게 되었는데, 처음에는 모델을 다루는 것이 어렵게만 느껴졌다. 특히 최종적으로 어떤 모델을 사용할지 정하는 것부터, 사용하기로 정한 모델에 맞게 학습과 추론 코드를 수정하는 것이 꽤 어려웠다. 그러나 최종적으로, 사용할 모델을 라우팅하는 것과 모델이 실제로 추론하는 것을 보자 성취감을 느낄 수 있었다.|과제 수행을 위해 모델의 파인튜닝을 하면서 시행착오를 많이 겪었다. 하지만 시행착오를 통해 배우는 부분도 많이 있었고, 고민한 문제를 해결하면서 성취감도 얻을 수 있어 의미 있는 경험이 된 것 같아 좋았다.|

---

### 8. 참고 문헌 및 출처
```
[1] G. Lai, Q. Xie, H. Liu, Y. Yang, and E. Hovy, “RACE: Large-scale ReAding Comprehension Dataset From Examinations,” *arXiv preprint* arXiv:1704.04683, 2017.

[2] Q. Xie, G. Lai, Z. Dai, and E. Hovy, “Large-scale Cloze Test Dataset Created by Teachers,” *arXiv preprint* arXiv:1711.03225, Aug. 2018. (Presented at EMNLP 2018)

[3] Gemma Team, A. Kamath, J. Ferret, S. Pathak, N. Vieillard, R. Merhej, S. Perrin, T. Matejovicova, A. Ramé, M. Rivière *et al.*, “Gemma 3 Technical Report,” *arXiv preprint* arXiv:2503.19786, Mar. 2025.

[4] Microsoft, A. Abouelenin, A. Ashfaq, A. Atkinson, H. Awadalla, N. Bach, J. Bao, A. Benhaim, M. Cai, V. Chaudhary *et al.*, “Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs,” *arXiv preprint* arXiv:2503.01743, Mar. 2025.

[5] E. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-Rank Adaptation of Large Language Models,” *International Conference on Learning Representations (ICLR)*, 2022.

[6] E. Zelikman, Y. Wu, J. Mu, and N. D. Goodman, “STaR: Bootstrapping Reasoning With Reasoning,” *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, pp. 15476–15488, 2022.

[7] T. G. Dietterich, “Ensemble Methods in Machine Learning,” in *Multiple Classifier Systems*, Berlin, Heidelberg: Springer, 2000, pp. 1–15. doi: 10.1007/3-540-45014-9_1.

[8] T. Heskes, “Selecting Weighting Factors in Logarithmic Opinion Pools,” *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 10, pp. 267-273, 1997.

[9] https://www.cs.cmu.edu/~glai1/data/race/

[10] https://www.cs.cmu.edu/~glai1/data/cloth/
```
