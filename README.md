# Emotional Chatbot
일상적이고 감정적인 대화를 하는 챗봇 설계

## 1. 목표
- 감정 Embedding을 단어와 함께 입력하여, 적절한 대답 문장을 생성하는 모델 작성
- 감정 분류 모델과 답변 생성 모델이 필요
- 신경망 종류에 따른 성능 비교

## 2. 구조
### 감정 분류 모델
- TextCNN : 문장을 입력하면, 1D CNN을 통해 감정을 분류
(reference : http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf)

### 대답 생성 모델
- Sequence-to-Sequence : LSTM으로 구성된 인코더와 디코더 존재, Attnetion mechanism 적용, inference에선 beam-search 방식 채택
- Transformer : Self-attention으로 구성된 인코더와 디코더 존재
(reference : https://github.com/Kyubyong/transformer)
- 비교를 위해 CNN이 적용되지 않은 Sequence-to-sequence 모델 (non_cnn) 포함

## 3. Prerequisite
- python = 3.6
- tensorflow == 1.15.2

## 4. 데이터 준비
- 'data' 폴더 상위 생성
- 아래 양식으로 작성
```
A : [질문 문장]
B : [대답 문장] [감정 태그]
```

## 5. 코드 실행
- script/hyperparam.py : 네트워크 세부 설정 변경 가능
    - make_data : True/False -> 데이터 초기 설정 여부
    - train : seq2seq/transformer/non_cnn/infer -> 훈련시킬 모델 설정 및 인퍼 여부
    - infer : seq2seq/transformer/non_cnn -> 인퍼 모델 설정
- script/main.py : 메인 스크립트
```
cd script
python main.py
```