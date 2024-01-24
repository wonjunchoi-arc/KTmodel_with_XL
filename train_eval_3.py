import os
import sys
import pandas as pd
import numpy as np
import json
import copy
import itertools
import tensorflow as tf
from KT.preprocess.data_utils import read_data,get_max_concepts,calStatistics,extend_multi_concepts,id_mapping,save_id2idx,train_test_split,save_dcur,generate_sequences, get_mask_tokens
import pickle
import glob
import matplotlib.pyplot as plt
import time
from KT.models.model_for_kt import TFTransfoXLModel,TFTransfoXLLMHeadModel,TFTransfoXLMLMHeadModel

from transformers import TransfoXLConfig
from tensorflow.keras.utils import register_keras_serializable
import datetime
import horovod.tensorflow as hvd


# Horovod 초기화
hvd.init()

# GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


config_xl = TransfoXLConfig(
    data = '/home/jun/workspace/KT/data/ednet/data.txt',
    dataset = 'wt103',
    d_embed=128,
    d_head = 32,
    d_model=128,
    mem_len=400,
    n_head=8,
    n_layer=6,
    batch_size = 60,
    tgt_len = 130,
    ext_len = 0,
    eval_tgt_len = 36,
    eos_token=2,
    num_c=123,
    mask_token=3,
    C_vocab_size=188,
    R_vocab_size = 2
)


def get_evalmask_token(R, mask_token,eos_token, mlm_probability=0.15):
    labels = tf.identity(R)
    
    # 확률 행렬 생성
    probability_matrix = tf.fill(tf.shape(labels), mlm_probability)
    
    # 특별 토큰에 대한 마스크 생성
    special_token_mask = tf.equal(R, eos_token)
    
    # 확률 행렬에 특별 토큰 마스크 적용
    probability_matrix = tf.where(special_token_mask, 0.0, probability_matrix)
    
    # 마스킹할 인덱스 생성
    masked_indices = tf.cast(tf.random.uniform(tf.shape(labels)) < probability_matrix, tf.bool)
    
    # 모든 마스킹된 인덱스 찾기
    all_masked_indices = tf.where(masked_indices)
    all_masked_indices = all_masked_indices[:, 0]
    
    # 마스킹할 인덱스가 없는 경우, 원래의 R 및 labels 반환
    if tf.size(all_masked_indices) == 0:
        return R, tf.fill(tf.shape(labels), -100)
    
    # 랜덤으로 선택된 마스킹된 인덱스
    random_index = tf.random.uniform(shape=(), minval=0, maxval=tf.size(all_masked_indices), dtype=tf.int32)
    selected_masked_index = tf.cast(tf.gather(all_masked_indices, random_index), dtype=tf.int32)
    
    # 랜덤으로 선택된 마스킹된 인덱스 이후의 모든 요소에 대한 마스크 생성
    extended_mask = tf.range(tf.size(labels)) > selected_masked_index
    extended_mask = tf.logical_and(extended_mask, ~special_token_mask)
    
    # 확장된 마스크 적용
    labels = tf.where(~extended_mask, -100, labels)
    R = tf.where(extended_mask, mask_token, R)

    return R, labels



ALL_KEYS = ["fold", "uid", "questions", "concepts", "responses", "timestamps",
            "usetimes", "selectmasks", "is_repeat", "qidxs", "rest", "orirow", "cidxs"]
ONE_KEYS = ["fold", "uid"]

total_df, effective_keys = read_data('/home/jun/workspace/KT/data/ednet/data.txt')

stares = []

if 'concepts' in effective_keys:
    max_concepts = get_max_concepts(total_df)
else:
    max_concepts = -1

oris, _, qs, cs, seqnum = calStatistics(total_df, stares, "original")
print("="*20)
print(
    f"original total interactions: {oris}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")


# questions ,concepts 값들의 숫자를 재정의 하여 0~ 나오도록 만들 면서 is_repeat값 처리
# 14_54_14 으로 된 concepts를 14,54,14 로 분리하고 기존의 response도 똑같이 확장처리
total_df_ex, effective_keys = extend_multi_concepts(total_df, effective_keys)
total_df, dkeyid2idx = id_mapping(total_df_ex)
dkeyid2idx["max_concepts"] = max_concepts

extends, _, qs, cs, seqnum = calStatistics(
    total_df, stares, "extend multi")
print("="*20)
print(
    f"after extend multi, total interactions: {extends}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")


#train test 분리
train_df, test_df = train_test_split(total_df, 0.2)

#transformer_xl dataset 만들기
train = {"qseqs": [], "cseqs": [], "masked_R": [], "labels": []} #"q_shift": [], "c_shift": [], "r_shift": []}
test = {"qseqs": [], "cseqs": [], "masked_R": [], "labels": []} #"q_shift": [], "c_shift": [], "r_shift": []}


# eos_token= 2 # config 파일로 조정할 수 있도록 수정 할 것!
# num_c =123 # config 파일로 조정할 수 있도록 수정 할 것!
for i, row in train_df.iterrows():
#use kc_id or question_id as input

    cseq_list=[int(_) for _ in row["concepts"].split(",")]
    cseq_list.append(config_xl.eos_token)
    train["cseqs"].append(cseq_list)

    qes_list=[int(_) for _ in row["questions"].split(",")]
    qes_list.append(config_xl.eos_token)
    train["qseqs"].append(qes_list)


    rseq_list=[(int(_)) for _ in row["responses"].split(",")]
    rseq_list.append(config_xl.eos_token)
    masked_R, labels = get_mask_tokens(rseq_list,config_xl.mask_token)
    train["masked_R"].append(masked_R)
    train["labels"].append(labels)
for i, row in test_df.iterrows():
    #use kc_id or question_id as input
    
        cseq_list=[int(_) for _ in row["concepts"].split(",")]
        cseq_list.append(config_xl.eos_token)   
        test["cseqs"].append(cseq_list)

        qes_list=[int(_) for _ in row["questions"].split(",")]
        qes_list.append(config_xl.eos_token)
        test["qseqs"].append(qes_list)


        rseq_list=[(int(_)) for _ in row["responses"].split(",")]
        rseq_list.append(config_xl.eos_token)
        masked_R, labels = get_evalmask_token(rseq_list, config_xl.mask_token, config_xl.eos_token)
        
        test["masked_R"].append(masked_R)
        test["labels"].append(labels)

#train
cseqs_list = list(itertools.chain(*train['cseqs']))
qseqs_list = list(itertools.chain(*train['qseqs']))
r_masked_list = tf.concat(train['masked_R'], axis=0)
labels = tf.concat(train['labels'], axis=0)

n_step = len(cseqs_list) // (config_xl.batch_size*config_xl.tgt_len)


sliced_cseqs = tf.slice(cseqs_list,[0],[n_step * config_xl.batch_size*config_xl.tgt_len])  
sliced_qseqs = tf.slice(qseqs_list,[0],[n_step * config_xl.batch_size*config_xl.tgt_len])  
sliced_r_mask = tf.slice(r_masked_list,[0],[n_step * config_xl.batch_size*config_xl.tgt_len]) 
sliced_labels = tf.slice(labels,[0],[n_step * config_xl.batch_size*config_xl.tgt_len]) 

count =len(sliced_cseqs)// (config_xl.batch_size*config_xl.tgt_len)

new_shape = (config_xl.batch_size, -1)  # 나머지 차원은 자동으로 계산됨

cseq_reshaped = tf.reshape(sliced_cseqs, new_shape)
r_mask_reshaped = tf.reshape(sliced_r_mask, new_shape)
labels_reshaped = tf.reshape(sliced_labels, new_shape)

cseq_transposed = tf.transpose(cseq_reshaped)
r_mask_transposed = tf.transpose(r_mask_reshaped)
labels_transposed = tf.transpose(labels_reshaped)


#test
test_cseqs_list = list(itertools.chain(*test['cseqs']))
test_qseqs_list = list(itertools.chain(*test['qseqs']))
test_r_masked_list = tf.concat(test['masked_R'], axis=0)
test_labels = tf.concat(test['labels'], axis=0)



# Work out how cleanly we can divide the dataset into bsz parts.
# 아래의 두 코드는   data 텐서에서 배치 크기 bsz로 깔끔하게 맞지 않는 추가 요소를 제거하는 것 배치에 띡 떨어지게
    
test_n_step = len(test_cseqs_list) // (config_xl.batch_size*config_xl.tgt_len)
print('n_step',test_n_step) # 

test_sliced_cseqs = tf.slice(test_cseqs_list,[0],[test_n_step * config_xl.batch_size*config_xl.tgt_len])  
test_sliced_qseqs = tf.slice(test_qseqs_list,[0],[test_n_step * config_xl.batch_size*config_xl.tgt_len])  
test_sliced_r_masked_seq = tf.slice(test_r_masked_list,[0],[test_n_step * config_xl.batch_size*config_xl.tgt_len]) 

test_sliced_labels = tf.slice(test_labels,[0],[test_n_step * config_xl.batch_size*config_xl.tgt_len])  

count =len(test_sliced_cseqs)// (config_xl.batch_size*config_xl.tgt_len)
print(count)

'''# 시작 위치와 슬라이싱할 크기 설정
begin = [0]  # 첫 번째 차원의 시작 위치는 0
size = [6]   # 첫 번째 차원에서 6개의 원소를 슬라이싱

# 데이터를 잘라내기 (tf.slice 사용)
sliced_data = tf.slice(data, begin, size)  '''

# Evenly divide the da
# ta across the bsz batches.

new_shape = (config_xl.batch_size, -1)  # 나머지 차원은 자동으로 계산됨

test_cseq_reshaped = tf.reshape(test_sliced_cseqs, new_shape)
test_r_masked_seq_reshaped = tf.reshape(test_sliced_r_masked_seq, new_shape)
test_labels_reshaped = tf.reshape(test_sliced_labels, new_shape)


test_cseq_reshaped = tf.transpose(test_cseq_reshaped)
test_r_masked_seq_reshaped = tf.transpose(test_r_masked_seq_reshaped)
test_labels_reshaped = tf.cast(tf.transpose(test_labels_reshaped),tf.int64)


train_dataset = tf.data.Dataset.from_tensor_slices(
    (cseq_transposed, r_mask_transposed, labels_transposed))
test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_cseq_reshaped, test_r_masked_seq_reshaped, test_labels_reshaped))


train_dataset =train_dataset.batch(config_xl.tgt_len)
test_dataset =test_dataset.batch(config_xl.tgt_len)


# GPU별 훈련을 위해 반으로 자름

train_first_half_dataset = train_dataset.take(1331 //2)
train_second_half_dataset = train_dataset.skip((1331 // 2) + (1331 % 2))

test_first_half_dataset = test_dataset.take(342//2)
test_second_half_dataset = test_dataset.skip((342 // 2) + (342 % 2))



@register_keras_serializable()
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)

        self.warmup_steps = tf.cast(warmup_steps,tf.float32)

    def __call__(self, step):
        step =tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
    
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)*hvd.size()
    
    def get_config(self):
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
            }
learning_rate = CustomSchedule(config_xl.d_model)


# 옵티마이저 정의 및 Horovod 래핑
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
optimizer = hvd.DistributedOptimizer(optimizer)


# 텐서보드  저장 경로 및 writer 
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = '/home/jun/workspace/KT/logs/gradient_tape/' + current_time + '/train'
test_log_dir = '/home/jun/workspace/KT/logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)



#평가지표 정의
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
  name='test_accuracy')
test_precision = tf.metrics.Precision()
test_recall = tf.metrics.Recall()
train_auc = tf.keras.metrics.AUC()
test_auc = tf.keras.metrics.AUC()



# 모델 정의
# 모델 생성
if hvd.rank() == 0:
    model0 = TFTransfoXLMLMHeadModel(config=config_xl)
  # GPU:0에서 사용할 첫 번째 모델
    # dataset0 = train_first_half_dataset       # 첫 번째 데이터셋
elif hvd.rank() == 1:
    model1 = TFTransfoXLMLMHeadModel(config=config_xl)
  # GPU:1에서 사용할 두 번째 모델
    # dataset1 = train_second_half_dataset       # 두 번째 데이터셋


# 훈련 스텝 정의
@tf.function
def train_step(model, data1,data2, target, mems, optimizer,first_batch):
    with tf.GradientTape() as tape:
        outputs = model(concepts=data1,responses=data2, labels=target, mems=mems)
        loss = outputs.loss
        mems = outputs.mems

        loss_mx = target != -100
        loss_value = loss[loss_mx]
        loss_value = tf.reshape(loss_value, [-1, config_xl.R_vocab_size])
        labels = target[loss_mx]

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=loss_value)
        predictions =tf.nn.softmax(loss_value)

        # batch_loss = tf.reduce_sum(loss) / valid_samples
        mean_loss = tf.reduce_mean(loss)
        # train_loss(loss)
        # train_accuracy(labels,loss_value)
        # train_auc(tf.one_hot(labels, depth=predictions.shape[1]), predictions)

    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape)

    gradients = tape.gradient(mean_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    #첫 배치에서 0gpu의 모델 및 옵티마이저 변수를 복제하여 동기화!  
    if first_batch :
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(optimizer.variables(), root_rank=0)

    return mems,mean_loss

def evaluate(model, mems, test_dataset):
    total_loss = 0.0
    num_batches = 0
    evaluation_metrics = []
    

    for test_data1, test_data2, test_target in test_dataset:
        outputs = model(concepts=test_data1, responses=test_data2, labels=test_target, mems=mems, training=False)
        loss = outputs.loss
        mems = outputs.mems

        loss_mx = test_target != -100
        loss_value = loss[loss_mx]
        loss_value = tf.reshape(loss_value, [-1, config_xl.R_vocab_size])
        labels = test_target[loss_mx]

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=loss_value)
        mean_loss = tf.reduce_mean(loss)

        # Update precision and recall metrics
        predicted_labels = tf.argmax(loss_value, axis=1)
        predictions =tf.nn.softmax(loss_value)

        
        test_auc(tf.one_hot(labels, depth=predictions.shape[1]), predictions)
        test_precision(labels, predicted_labels)
        test_recall(labels, predicted_labels)

        test_accuracy(labels, loss_value)
        test_loss(loss)
        
        
        precision = test_precision.result().numpy()
        recall = test_recall.result().numpy()
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

        evaluation_metrics.append(test_accuracy.result().numpy())

        total_loss += mean_loss.numpy()
        num_batches += 1

        
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=num_batches)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=num_batches)
            tf.summary.scalar('precision', test_precision.result(), step=num_batches)
            tf.summary.scalar('recall', test_recall.result(), step=num_batches)
            tf.summary.scalar('f1_score', f1_score, step=num_batches)
            tf.summary.scalar('auc', test_auc.result(), step=num_batches)

    # 평균 정밀도, 재현율, F1 점수를 계산합니다.
    average_precision = test_precision.result().numpy()
    average_recall = test_recall.result().numpy()
    average_f1_score = 2 * (average_precision * average_recall) / (average_precision + average_recall + 1e-7)

    average_metric = sum(evaluation_metrics) / len(evaluation_metrics)
    average_loss = total_loss / num_batches

    return average_loss, average_metric, average_precision, average_recall, average_f1_score



loss_values = []
num_batches = 0

for epoch in range(1):
    start = time.time()
    total_loss = 0.0
    if hvd.rank() == 0:
        mems0 = None              # 첫 번째 모델의 메모리 상태
        # dataset0 = train_first_half_dataset
        for data1, data2, target in train_first_half_dataset:
            print('gpu0 data1',data1)

            mems0, loss_value = train_step(model0, data1,data2, target, mems0, optimizer,num_batches==0)
            num_batches += 1
            # print('loss_value',loss_value)
            total_loss += loss_value.numpy()
            if num_batches % 100 == 0:
                loss_values.append(loss_value.numpy())
                print(f'gpu0 Epoch {epoch + 1} Batch {num_batches} Loss {loss_value.numpy()}')
    elif hvd.rank() == 1:
        mems1 = None              # 첫 번째 모델의 메모리 상태
        # dataset1 = train_second_half_dataset
        for data1,data2, target in train_second_half_dataset:
            print('gpu1 data1',data1)
            mems1, loss_value = train_step(model1, data1,data2, target, mems1, optimizer,num_batches==0)
            total_loss += loss_value.numpy()
            num_batches += 1

            if num_batches % 100 == 0:
                print(f'gpu1 Epoch {epoch + 1} Batch {num_batches} Loss {loss_value.numpy()}')
            # with train_summary_writer.as_default():
            #     tf.summary.scalar('loss', train_loss.result(), step=num_batches)
            #     tf.summary.scalar('accuracy', train_accuracy.result(), step=num_batches)
            #     tf.summary.scalar('auc', train_auc.result(), step=num_batches) 
               

        # save_path = f'/home/jun/workspace/KT/data/model_save/model_epoch_{epoch+1}'

        # model1.save(save_path)
    # Calculate and print the average loss for the epoch
    average_loss = total_loss / num_batches
    print(f'Epoch {epoch + 1} Average Loss: {average_loss}')

    
    # test_mems = None
    # test_loss0,test_acc0,average_precision, average_recall, average_f1_score = evaluate(model1,test_mems, test_dataset)
    # print(f'Test Loss on First Half Dataset after Epoch {epoch + 1}: {test_loss0}')


# horovodrun -np 2 -H localhost:2 python /home/jun/workspace/KT/train_eval_2.py 실행 명령어

#방법 호로보드 제외하고 단일 gpu로 한번 만들어보자!!