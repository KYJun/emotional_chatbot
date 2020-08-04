### Text CNN module scripts

- train_cnn.py
```
cnn = Cnn_Graph(vocab_size, maxlen, is_training)
```
Cnn Graph Class
hyperparam.py에서 변수를 수정 가능하다.
is_training=True인 경우, batch iterator를 통해 데이터를 전달하며 optimizer를 통한 모델 업데이트가 가능하다.
is_training=False인 경우, placeholder에 feed_dict로 데이터를 전달하여 test 혹은 inference가 가능하다.

```
train_cnn()
```
**main train function for cnn**
hyperparam.py에서 변수를 수정 가능하다.
지정한 logdir에 checkpoint가 저장된다.

- test_cnn.py
```
test_cnn()
```
터미널 창에서 입력한 문장의 tag 예측
