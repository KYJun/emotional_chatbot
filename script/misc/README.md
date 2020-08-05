### Supplementary Scripts

: data 관리를 담당하는 함수를 정리하였다. create_data( )를 제외하고 단독으로 쓰이지 않는다.

#### data_prep.py

data 예시는 다음과 같다.

```
A: [문장]
B: [문장], [감정 태그]
```

```
input_data, target_data, tag, word_list = read_raw_data(filename)
```
text 파일을 입력하면, 한국어 단어 이외의 불필요한 기호를 제거한다. 
array 형식의 human_input, ai_output, sentence_tag, vocabulary 단어 목록을 출력한다. 

```
vocab, rev_vocab = build_vocab(word_list)
```
단어 목록을 입력하면, 빈도수 기준으로 일련번호를 부여한다.
UNK, PAD, STR, END 토큰이 단어 사전에 추가된다.
단어 사전의 크기는 ../hyperparams.py에 지정된 vocab_size에 맞춰 형성된다.
단어와 일련번호의 조합을 dict 형식의 vocab과 list 형식의 rev_vocab으로 출력한다.

```
result = get_token_id(tokens, vocab, add_ss)
```
각 문장의 단어를 일련번호로 치환하여 출력한다.
add_ss=True일 때, STR / END 토큰이 문장 앞뒤에 추가된다.

```
X, Y = build_input(x_data, y_data, vocab)
```
전체 human_input과 ai_outpt을 padding이 포함된 일련번호 array로 변환한다.
이때, X와 Y의 최대 문장 길이는 ../hyperparams.py에 지정된 maxlen을 따라간다.

```
save_pickle(data, filepath)
file = load_pickle(filepath)
```
주어진 파일을  ".pkl" 확장자로 저장하고 불러온다.

```
create_data()
```
**main data preparation function**
../hyperparams.py에 지정된 train data 파일을 읽어 padded index array로 변환하고 저장하며,
vocab과 rev_vocab 파일도 같은 폴더에 저장한다.

```
vocab_size = get_vocabsize()
maxlen = get_maxlen()
```
vocab_size 혹은 maxlen이 None으로 지정된 경우, 해당 hyperparameter를 저장된 파일에서 불러온다.

```
input_data, target_data = load_train_data(model, filepath)
```
create_data에서 저장된 파일을 각 model에 맞게 불러낸다.
cnn의 경우, human_input, ai_output을 합쳐 input data가 되고, 각각에 문장에 맞는 tag가 target data가 된다.
transformer의 경우, human_input이 input data가 되고, ai output이 target data가 된다.

```
iter = batch_tf_data(model, filepath)
```
주어진 모델에 맞게 저장된 파일을 불러와 tf.data를 활용하여 batch iterator를 만든다.
cnn의 경우, input (human_input + ai_output) 과 target (sentence_tag*2)의 batch iterator가 만들어진다.
transformer의 경우, input (human_input), target (ai_output), tag (predicted_tag)의 batch iterator가 만들어진다.
