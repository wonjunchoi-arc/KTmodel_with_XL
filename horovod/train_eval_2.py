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
    batch_size = 40,
    tgt_len = 120,
    ext_len = 0,
    eval_tgt_len = 36,
    eos_token=2,
    num_c=123,
    mask_token=3,
    C_vocab_size=188,
    R_vocab_size = 2
)



# train_path = '/home/jun/workspace/KT/data/ednet/train.csv'

# test_path = '/home/jun/workspace/KT/data/ednet/test.csv'

# if not os.path.exists(train_path):
    
#     # read txt dataset to dataframe
#     ALL_KEYS = ["fold", "uid", "questions", "concepts", "responses", "timestamps",
#                 "usetimes", "selectmasks", "is_repeat", "qidxs", "rest", "orirow", "cidxs"]
#     ONE_KEYS = ["fold", "uid"]
    
#     total_df, effective_keys = read_data(config_xl.data)
    
#     stares = []
    
#     if 'concepts' in effective_keys:
#         max_concepts = get_max_concepts(total_df)
#     else:
#         max_concepts = -1
    
#     oris, _, qs, cs, seqnum = calStatistics(total_df, stares, "original")
#     print("="*20)
#     print(
#         f"original total interactions: {oris}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    
    
#     # questions ,concepts 값들의 숫자를 재정의 하여 0~ 나오도록 만들 면서 is_repeat값 처리
#     total_df, effective_keys = extend_multi_concepts(total_df, effective_keys)
#     total_df, dkeyid2idx = id_mapping(total_df)
#     dkeyid2idx["max_concepts"] = max_concepts
    
#     extends, _, qs, cs, seqnum = calStatistics(
#         total_df, stares, "extend multi")
#     print("="*20)
#     print(
#         f"after extend multi, total interactions: {extends}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    
    
#     #train test 분리
#     train_df, test_df = train_test_split(total_df, 0.2)
    
#     train_df.to_csv(train_path,index=False)
#     test_df.to_csv(test_path,index=False)
    


# def make_eval_mask(R,mask_token,mlm_probability=0.15):
#     labels = tf.identity(R)
        
#     # 기존 probability_matrix 생성
#     probability_matrix = tf.fill(tf.shape(labels), mlm_probability)
    
#     # special_token_mask 생성
#     special_token_mask = tf.equal(R, 2)
    
#     # probability_matrix에 special_token_mask 적용
#     probability_matrix = tf.where(special_token_mask, 0.0, probability_matrix)
    
#     # 마스킹할 인덱스 생성
#     masked_indices = tf.cast(tf.random.uniform(tf.shape(labels)) < probability_matrix, tf.bool)
    
#     # 마스킹된 모든 인덱스 찾기
#     all_masked_indices = tf.where(masked_indices)[:,0]
    
#     # 마스킹된 인덱스 중 랜덤으로 특정값 선택
#     random_index = tf.random.uniform(shape=(), minval=0, maxval=tf.size(all_masked_indices), dtype=tf.int32)
#     selected_masked_index = tf.cast(tf.gather(all_masked_indices, random_index),dtype=tf.int32)
    
        
#     # Create a mask for all elements after the randomly selected masked index
#     extended_mask = tf.range(tf.size(labels)) > selected_masked_index
#     extended_mask = tf.logical_and(extended_mask, ~special_token_mask)
    
#     # Apply extended mask
#     labels = tf.where(~extended_mask, -100, labels)
#     R = tf.where(extended_mask, mask_token, R)

#     return R, labels
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




# class TrainDataGenerator:
#     def __init__(self, train_path, bsz, bptt, eos_token, mask_token, ext_len=None):
#         self.train_path = train_path.decode('utf-8') if isinstance(train_path, bytes) else train_path
#         self.bsz = bsz
#         self.bptt = bptt
#         self.eos_token = eos_token
#         self.mask_token = mask_token
#         self.ext_len = ext_len if ext_len is not None else 0
#         self.n_step = None
#         self.train_df = pd.read_csv(self.train_path)


def train_gen(data,eos_token,bsz,bptt,mask_token):


    ALL_KEYS = ["fold", "uid", "questions", "concepts", "responses", "timestamps",
                "usetimes", "selectmasks", "is_repeat", "qidxs", "rest", "orirow", "cidxs"]
    ONE_KEYS = ["fold", "uid"]
    
    total_df, effective_keys = read_data(data)
    
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
    total_df, effective_keys = extend_multi_concepts(total_df, effective_keys)
    total_df, dkeyid2idx = id_mapping(total_df)
    dkeyid2idx["max_concepts"] = max_concepts
    
    extends, _, qs, cs, seqnum = calStatistics(
        total_df, stares, "extend multi")
    print("="*20)
    print(
        f"after extend multi, total interactions: {extends}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    
    
    #train test 분리
    train_df, test_df = train_test_split(total_df, 0.2)
    
    #transformer_xl dataset 만들기
    dori = {"qseqs": [], "cseqs": [], "rseqs": []} #"q_shift": [], "c_shift": [], "r_shift": []}

    # eos_token= 2 # config 파일로 조정할 수 있도록 수정 할 것!
    # num_c =123 # config 파일로 조정할 수 있도록 수정 할 것!
    for i, row in train_df.iterrows():
    #use kc_id or question_id as input
    
        cseq_list=[int(_) for _ in row["concepts"].split(",")]
        cseq_list.append(eos_token)
        dori["cseqs"].append(cseq_list)

        qes_list=[int(_) for _ in row["questions"].split(",")]
        qes_list.append(eos_token)
        dori["qseqs"].append(qes_list)


        rseq_list=[(int(_)) for _ in row["responses"].split(",")]
        rseq_list.append(eos_token)
        dori["rseqs"].append(rseq_list)

        # c_shift_list=[int(_) for _ in row["concepts"].split(",")]
        # c_shift_list.append(eos_token)
        # dori["c_shift"].append(c_shift_list)


        # r_shift_list=[int(_) for _ in row["responses"].split(",")]
        # r_shift_list.append(eos_token)
        # dori["r_shift"].append(r_shift_list)

    '''
    딕셔너리의 각 값마다 끝에 eos 토큰 삽입
    rseqs에는 num_c 곱하여 cseqs 더할 값 만들기

    '''
    cseqs_list = list(itertools.chain(*dori['cseqs']))
    qseqs_list = list(itertools.chain(*dori['qseqs']))
    rseqs_list = list(itertools.chain(*dori['rseqs']))
    # c_shift_list = list(itertools.chain(*dori['c_shift']))
    # r_shift_list = list(itertools.chain(*dori['r_shift']))


    
    
    
    # Work out how cleanly we can divide the dataset into bsz parts.
    # 아래의 두 코드는   data 텐서에서 배치 크기 bsz로 깔끔하게 맞지 않는 추가 요소를 제거하는 것 배치에 띡 떨어지게
    n_step = len(cseqs_list) // (bsz*bptt)
    print('n_step',n_step) # 
    
    sliced_cseqs = tf.slice(cseqs_list,[0],[n_step * bsz*bptt])  
    sliced_qseqs = tf.slice(qseqs_list,[0],[n_step * bsz*bptt])  
    sliced_rseqs = tf.slice(rseqs_list,[0],[n_step * bsz*bptt]) 


    count =len(sliced_cseqs)// (bsz*bptt)
    n_step = count
    '''# 시작 위치와 슬라이싱할 크기 설정
    begin = [0]  # 첫 번째 차원의 시작 위치는 0
    size = [6]   # 첫 번째 차원에서 6개의 원소를 슬라이싱

    # 데이터를 잘라내기 (tf.slice 사용)
    sliced_data = tf.slice(data, begin, size)  '''

    # Evenly divide the da
    # ta across the bsz batches.

    new_shape = (bsz, -1)  # 나머지 차원은 자동으로 계산됨

    cseq_reshaped = tf.reshape(sliced_cseqs, new_shape)
    rseq_reshaped = tf.reshape(sliced_rseqs, new_shape)
    # c_shift_reshaped = tf.reshape(sliced_c_shift, new_shape)
    # r_shift_reshaped = tf.reshape(sliced_r_shift, new_shape)
    # data_transposed = tf.transpose(data_reshaped)
    # print('interaction_reshaped',interaction_reshaped.shape)
    split_num = 2 #GPU num


    # first_half, second_half = tf.split(data, num_or_size_splits=split_num, axis=1)

    # n_batch = (n_step + self.bptt - 1) // self.bptt

    for i in range(0, len(rseq_reshaped[1]) - 1, bptt):
        
        seq_len = min(bptt, rseq_reshaped.shape[1] - 1 - i) # # i값이 103227020를 넘지 않는 이상 seq_len = 70


        end_idx = i + seq_len # 70,71,72,73,74......
        beg_idx = max(0, i ) # 0,1,2,3,4,5
        ''' 아래 처럼 첫번째 차원을 자르는 이류
        로,또,1,등,당,첨 = > 로,또,1    => 로, 등
                        등,당,첨         또, 당
                                        1, 첨
        '''

        C = cseq_reshaped[:,beg_idx:end_idx] # self.data[:,0:70],[:,1:71] ~
        R = rseq_reshaped[:,beg_idx:end_idx] # self.data[:,0:70],[:,1:71] ~
        # label = rseq_reshaped[:,beg_idx:end_idx] # self.data[:,0:70],[:,1:71] ~
        # Query = c_shift_reshaped[:,i+1:i+1+seq_len] # self.data[:,0:70],[:,1:71] ~
        # r_shift = r_shift_reshaped[:,i+1:i+1+seq_len]

        # second_half_data = second_half[:,beg_idx:end_idx] # self.data[:,0:70],[:,1:71] ~
        # second_half_target = second_half[:,i+1:i+1+seq_len]
        '''
        여기서 원하는 값을 마스킹 해주도록 하자!
        '''
        masked_R, labels = get_mask_tokens(R,mask_token)


        if i + bptt < len(rseq_reshaped[1]) - 1:
            yield C, masked_R,labels
        # yield second_half_data, second_half_target

    # def get_n_step(self):
    #     return self.n_step


# class TestDataGenerator:
#     def __init__(self, test_path, bsz, bptt, eos_token, mask_token, ext_len=None):
#         self.test_path = test_path.decode('utf-8') if isinstance(test_path, bytes) else test_path
#         self.bsz = bsz
#         self.bptt = bptt
#         self.eos_token = eos_token
#         self.mask_token = mask_token
#         self.ext_len = ext_len if ext_len is not None else 0
#         self.n_step = None
#         self.test_df = pd.read_csv(self.test_path)

def test_gen(data,eos_token,bsz,bptt,mask_token):
    
    ALL_KEYS = ["fold", "uid", "questions", "concepts", "responses", "timestamps",
                "usetimes", "selectmasks", "is_repeat", "qidxs", "rest", "orirow", "cidxs"]
    ONE_KEYS = ["fold", "uid"]
    
    total_df, effective_keys = read_data(data)
    
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
    total_df, effective_keys = extend_multi_concepts(total_df, effective_keys)
    total_df, dkeyid2idx = id_mapping(total_df)
    dkeyid2idx["max_concepts"] = max_concepts
    
    extends, _, qs, cs, seqnum = calStatistics(
        total_df, stares, "extend multi")
    print("="*20)
    print(
        f"after extend multi, total interactions: {extends}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    
    
    #train test 분리
    train_df, test_df = train_test_split(total_df, 0.2)

    #transformer_xl dataset 만들기
    dori = {"qseqs": [], "cseqs": [], "r_mask_seqs": [], "labels":[]} #"q_shift": [], "c_shift": [], "r_shift": []}


    # eos_token= 2 # config 파일로 조정할 수 있도록 수정 할 것!
    # num_c =123 # config 파일로 조정할 수 있도록 수정 할 것!
    for i, row in test_df.iterrows():
    #use kc_id or question_id as input
    
        cseq_list=[int(_) for _ in row["concepts"].split(",")]
        cseq_list.append(eos_token)   
        dori["cseqs"].append(cseq_list)

        qes_list=[int(_) for _ in row["questions"].split(",")]
        qes_list.append(eos_token)
        dori["qseqs"].append(qes_list)


        rseq_list=[(int(_)) for _ in row["responses"].split(",")]
        rseq_list.append(eos_token)
        masked_R, labels = get_evalmask_token(rseq_list, mask_token, eos_token)
        
        dori["r_mask_seqs"].append(masked_R)
        dori["labels"].append(labels)

        # c_shift_list=[int(_) for _ in row["concepts"].split(",")]
        # c_shift_list.append(eos_token)
        # dori["c_shift"].append(c_shift_list)


        # r_shift_list=[int(_) for _ in row["responses"].split(",")]
        # r_shift_list.append(eos_token)
        # dori["r_shift"].append(r_shift_list)

    '''
    딕셔너리의 각 값마다 끝에 eos 토큰 삽입
    rseqs에는 num_c 곱하여 cseqs 더할 값 만들기

    '''
    cseqs_list = list(itertools.chain(*dori['cseqs']))
    qseqs_list = list(itertools.chain(*dori['qseqs']))
    r_masked_list = tf.concat(dori['r_mask_seqs'], axis=0)
    labels = tf.concat(dori['labels'], axis=0)


    
    # Work out how cleanly we can divide the dataset into bsz parts.
    # 아래의 두 코드는   data 텐서에서 배치 크기 bsz로 깔끔하게 맞지 않는 추가 요소를 제거하는 것 배치에 띡 떨어지게
        
    n_step = len(cseqs_list) // (bsz*bptt)
    print('n_step',n_step) # 
    
    sliced_cseqs = tf.slice(cseqs_list,[0],[n_step * bsz*bptt])  
    sliced_qseqs = tf.slice(qseqs_list,[0],[n_step * bsz*bptt])  
    sliced_r_masked_seq = tf.slice(r_masked_list,[0],[n_step * bsz*bptt]) 
    
    sliced_labels = tf.slice(labels,[0],[n_step * bsz * bptt])  
    
    count =len(sliced_cseqs)// (bsz* bptt)
    print(count)
    
    '''# 시작 위치와 슬라이싱할 크기 설정
    begin = [0]  # 첫 번째 차원의 시작 위치는 0
    size = [6]   # 첫 번째 차원에서 6개의 원소를 슬라이싱

    # 데이터를 잘라내기 (tf.slice 사용)
    sliced_data = tf.slice(data, begin, size)  '''

    # Evenly divide the da
    # ta across the bsz batches.

    new_shape = (bsz, -1)  # 나머지 차원은 자동으로 계산됨
    
    cseq_reshaped = tf.reshape(sliced_cseqs, new_shape)
    r_masked_seq_reshaped = tf.reshape(sliced_r_masked_seq, new_shape)
    labels_reshaped = tf.reshape(sliced_labels, new_shape)
    split_num = 2 #GPU num
    
    
    # first_half, second_half = tf.split(data, num_or_size_splits=split_num, axis=1)
    
    # n_batch = (n_step + self.bptt - 1) // self.bptt
    
    for i in range(0, len(cseq_reshaped[1]) - 1, bptt):
        
        seq_len = min(bptt, cseq_reshaped.shape[1] - 1 - i) # # i값이 103227020를 넘지 않는 이상 seq_len = 70
    
    
        end_idx = i + seq_len # 70,71,72,73,74......
        beg_idx = max(0, i) # 0,1,2,3,4,5
        ''' 아래 처럼 첫번째 차원을 자르는 이류
        로,또,1,등,당,첨 = > 로,또,1    => 로, 등
                        등,당,첨         또, 당
                                        1, 첨
        '''
    
        C = cseq_reshaped[:,beg_idx:end_idx] # self.data[:,0:70],[:,1:71] ~
        masked_R = r_masked_seq_reshaped[:,beg_idx:end_idx] # self.data[:,0:70],[:,1:71] ~
        labels = labels_reshaped[:,beg_idx:end_idx] # self.data[:,0:70],[:,1:71] ~


        if i + bptt < cseq_reshaped.shape[1] - 1:
            yield C, masked_R,labels
        # yield second_half_data, second_half_target

    # def get_n_step(self):
    #     return self.n_step


# train_gen_obj = TrainDataGenerator(train_path,
#      config_xl.batch_size,
#      config_xl.tgt_len,
#      config_xl.eos_token,
#      config_xl.mask_token
# )

# test_gen_obj = TestDataGenerator(
#     test_path,
#      config_xl.batch_size,
#      config_xl.tgt_len,
#      config_xl.eos_token,
#      config_xl.mask_token
# )


train_dataset = tf.data.Dataset.from_generator(
     train_gen,
     output_signature=(
         tf.TensorSpec(shape=None, dtype=tf.int64),
         tf.TensorSpec(shape=None, dtype=tf.int64),
         tf.TensorSpec(shape=None, dtype=tf.int64),
         ),args=(config_xl.data,config_xl.eos_token,config_xl.batch_size,config_xl.tgt_len,config_xl.mask_token)
         )
test_dataset = tf.data.Dataset.from_generator(
     test_gen,
     output_signature=(
         tf.TensorSpec(shape=None, dtype=tf.int64),
         tf.TensorSpec(shape=None, dtype=tf.int64),
         tf.TensorSpec(shape=None, dtype=tf.int64),
         ),args=(config_xl.data,config_xl.eos_token,config_xl.batch_size,config_xl.tgt_len,config_xl.mask_token)
         )


train_count = 0
test_count = 0
for i in train_dataset:
    train_count += 1
for i in test_dataset:
    test_count +=1

train_first_half_dataset = train_dataset.take(train_count //2)
train_second_half_dataset = train_dataset.skip((train_count // 2) + (train_count % 2))

test_first_half_dataset = test_dataset.take(342//2)
test_second_half_dataset = test_dataset.skip((test_count // 2) + (test_count % 2))



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



# CustomSchedule 및 모델 정의는 이전과 동일하게 유지합니다.
# ...

# 옵티마이저 정의 및 Horovod 래핑
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
optimizer = hvd.DistributedOptimizer(optimizer)

# 모델 정의

# 모델 및 데이터셋 생성
if hvd.rank() == 0:
    model0 = TFTransfoXLMLMHeadModel(config=config_xl)
  # GPU:0에서 사용할 첫 번째 모델
    # dataset0 = train_first_half_dataset       # 첫 번째 데이터셋
elif hvd.rank() == 1:
    model1 = TFTransfoXLMLMHeadModel(config=config_xl)
  # GPU:1에서 사용할 두 번째 모델
    # dataset1 = train_second_half_dataset       # 두 번째 데이터셋


from tensorflow import keras
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
        label = tf.reshape(labels, [-1])    

        # valid_samples = tf.reduce_sum(tf.cast(loss_mx, tf.float32))

        # tf.print('loss_mx',loss_mx)
        # tf.print('valid_samples',valid_samples)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=loss_value)

        # batch_loss = tf.reduce_sum(loss) / valid_samples
        mean_loss = tf.reduce_mean(loss)

    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape)

    gradients = tape.gradient(mean_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if first_batch:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(optimizer.variables(), root_rank=0)

    return mems,mean_loss

def evaluate(model,mems, test_dataset):
    total_loss = 0.0
    num_batches = 0
    evaluation_metrics = []  # Store evaluation metrics here (e.g., accuracy, F1-score, etc.

    for test_data1, test_data2, test_target in test_dataset:
        outputs = model(concepts=test_data1, responses=test_data2, labels=test_target, mems=mems)  # Use mems=None for evaluation
        loss = outputs.loss
        mems = outputs.mems

        loss_mx = test_target != -100
        loss_value = loss[loss_mx]
        loss_value = tf.reshape(loss_value, [-1, config_xl.R_vocab_size])
        labels = test_target[loss_mx]
        

        # Compute any additional evaluation metrics based on your problem (e.g., accuracy)
        # For example, if your model returns logits, you can calculate accuracy as follows:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=loss_value)
        mean_loss = tf.reduce_mean(loss)

        # print('loss_value',loss_value)
        predicted_labels = tf.argmax(loss_value, axis=-1)
        # print('predicted_labels',predicted_labels)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, labels), tf.float32))
        # print('accuracy',accuracy)
        # Append the evaluation metric (e.g., accuracy) to the list
        evaluation_metrics.append(accuracy.numpy())

        total_loss += mean_loss
        num_batches += 1
        # print('num_batches',num_batches)
    # Calculate the average evaluation metric (e.g., accuracy) over all batches
    average_metric = sum(evaluation_metrics) / len(evaluation_metrics)
    average_loss = total_loss / num_batches

    return average_loss, average_metric


loss_values = []  # 각 배치의 손실 값을 저장할 리스트

for epoch in range(3):
    start = time.time()
    num_batches = 0
    total_loss = 0.0
    if hvd.rank() == 0:
        mems0 = None              # 첫 번째 모델의 메모리 상태
        # dataset0 = train_first_half_dataset
        dataset0 = iter(train_first_half_dataset)
        for data1, data2, target in dataset0:
            mems0, loss_value = train_step(model0, data1,data2, target, mems0, optimizer,num_batches==0)
            # print('loss_value',loss_value)
            num_batches += 1
            total_loss += loss_value.numpy()
            if num_batches % 100 == 0:
                loss_values.append(loss_value.numpy())
                print(f'Epoch {epoch + 1} Batch {num_batches} Loss {loss_value.numpy()}')
    elif hvd.rank() == 1:
        mems1 = None              # 첫 번째 모델의 메모리 상태
        # dataset1 = train_second_half_dataset
        dataset1 = iter(train_second_half_dataset)
        for data1,data2, target in dataset1:
            mems1, loss_value = train_step(model1, data1,data2, target, mems1, optimizer,num_batches==0)
            num_batches += 1
            total_loss += loss_value.numpy()
            if num_batches % 100 == 0:
                print(f'Epoch {epoch + 1} Batch {num_batches} Loss {loss_value.numpy()}')
                
                loss_values.append(loss_value.numpy())  # 여기서 손실 값을 저장

        # save_path = f'/home/jun/workspace/KT/data/model_save/model_epoch_{epoch+1}'

        # model1.save(save_path)
    # Calculate and print the average loss for the epoch
    average_loss = total_loss / num_batches
    print(f'Epoch {epoch + 1} Average Loss: {average_loss}')
# import csv
# with open('/home/jun/workspace/KT/loss/loss_values.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Batch', 'Loss'])
#     for i, loss in enumerate(loss_values, start=1):
#         writer.writerow([i * 100, loss])  # Assum
# from tensorflow.keras.models import load_model

# model_path = '/path/to/saved/model'

# # Load the model
# loaded_model = load_model(model_path)
# print("Model loaded successfully")
# test_mems = None  
# test_loss, test_acc = evaluate(loaded_model,test_mems, test_dataset)

# print('test_loss',test_loss)
# print('test_acc',test_acc)
    
    # # Perform testing after each epoch (you can replace this with your evaluation logic)
    # if hvd.rank() == 0:
        
    # elif hvd.rank() == 1:
        
    
    # # Calculate and print the time taken for the epoch
    # end = time.time()
    # print(f'Epoch {epoch + 1} Time: {end - start} seconds')



# horovodrun -np 2 -H localhost:2 python /home/jun/workspace/KT/train_eval_2.py 실행 명령어

#방법 호로보드 제외하고 단일 gpu로 한번 만들어보자!!