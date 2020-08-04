### Transformer Module scripts

: Transformer 함수 및 전체 graph, 결과 출력 스크립트

- transformer_module.py
    - transformer에서 사용할 함수 모음
    - normalize : layer normalization
    - embedding : 단어 embedding
    - positional encoding
    - multi-head attention
    - feedforward 
    - label_smoothing : one-hot이 아닌 probabilistic label 부여
    
- **train_transformer.py**
    - Tr_Graph(is_training, vocab_size, maxlen)
    : is_training=True시 모델 훈련 가능 및 batch iterator를 통한 data feeding 활성화
    - train_transformer( )
    : human_input, ai_output, 앞서 훈련한 textcnn에서 예측한 sentence_tag 기반으로 transformer 훈련
    
- **eval.py**
    - eval( ) 
    : 주어진 test 파일에서의 human input 문장을 textcnn으로 예측한 tag와 함께 transformer를 통해 가능한 답 출력
