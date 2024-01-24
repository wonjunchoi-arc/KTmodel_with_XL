from KT.models.model_for_kt import TFTransfoXLModel,TFTransfoXLLMHeadModel,TFTransfoXLMLMHeadModel
from transformers import TransfoXLConfig
import pandas as pd
import tensorflow as tf
from KT.preprocess.data_utils import  get_mask_tokens
from pathlib import Path


config_xl = TransfoXLConfig(
    data = '/home/jun/workspace/KT/data/ednet/data.txt',
    dataset = 'wt103',
    d_embed=128,
    d_head = 32,
    d_model=128,
    mem_len=400,
    n_head=8,
    n_layer=6,
    batch_size = 65,
    tgt_len = 140,
    ext_len = 0,
    eval_tgt_len = 36,
    eos_token=2,
    num_c=123,
    mask_token=3,
    C_vocab_size=188,
    R_vocab_size = 2
)

__version__="0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent
model = TFTransfoXLMLMHeadModel(config=config_xl)
with open(f"{BASE_DIR}/trained_pipline-{__version__}/my_checkpoint",'rb')as f:
    model.load_weights(f)


# csv 파일을 입력으로 받아서 concepts열과 reponse열을 선택하고 concepts를 쭉 나열 한 뒤 학생이 현재 지도에 맞춰 푼 문제를 response라 하여 제공 
#predict는 학생이 아직 풀지 않은 즉 마스킹 된 값들로 만들어서 예측할 수 있도록 하자 
def predict_pipline(path,uid):
        
    csv_file_path = path
    df= pd.read_csv(csv_file_path)

    question_list = df.loc[df['uid'] == uid, 'questions']
    concepts_list = df.loc[df['uid'] == uid, 'concepts']
    responses_list = df.loc[df['uid'] == uid, 'responses']

    question_list = [int(x) for x in question_list[0].split(',')]
    concepts_list = [int(x) for x in concepts_list[0].split(',')]
    responses_list = [int(x) for x in responses_list[0].split(',')]

    masked_R, labels = get_mask_tokens(responses_list,config_xl.mask_token) #실제 데이터에서는 알아서 mask되어서 들어오겠지



    # 여기서 읽어온 데이터를 딕셔너리키 가지고 변환해서 ip mapping 하고 

    n_step = len(concepts_list) // (config_xl.tgt_len)

    sliced_qseqs = tf.slice(question_list,[0],[n_step *config_xl.tgt_len])  
    sliced_cseqs = tf.slice(concepts_list,[0],[n_step *config_xl.tgt_len])  
    sliced_masked_R = tf.slice(masked_R,[0],[n_step *config_xl.tgt_len]) 
    sliced_labels = tf.slice(labels,[0],[n_step *config_xl.tgt_len]) 


    new_shape = (config_xl.batch_size, -1)  # 나머지 차원은 자동으로 계산됨

    qseq_reshaped = tf.reshape(sliced_qseqs, new_shape)
    cseq_reshaped = tf.reshape(sliced_cseqs, new_shape)
    masked_R_reshaped = tf.reshape(sliced_masked_R, new_shape)
    labels_reshaped = tf.reshape(sliced_labels, new_shape)


    cseq_transposed = tf.transpose(cseq_reshaped)
    masked_R_transposed = tf.transpose(masked_R_reshaped)
    labels_transposed = tf.transpose(labels_reshaped)

    predict_dataset = tf.data.Dataset.from_tensor_slices(
    (cseq_transposed, masked_R_transposed,labels_transposed))

    predict_dataset =predict_dataset.batch(config_xl.tgt_len)

    mems =None
    predictions = []
    for test_data1, test_data2, test_target in predict_dataset:
            outputs = model(concepts=test_data1, responses=test_data2, labels=test_target, mems=mems)
            loss = outputs.loss
            mems = outputs.mems
            
            reshape = tf.reshape(loss, [-1, config_xl.R_vocab_size])
            predicted_labels = tf.argmax(reshape, axis=1)
            predictions.append(predicted_labels.numpy().tolist())

    flattened_list = [item for sublist in predictions for item in sublist]

    df = pd.DataFrame({'Question':sliced_qseqs,'Concepts':sliced_cseqs,'Responses':flattened_list})

    df.to_csv(f'/home/jun/workspace/KT/data/ednet/predict/{uid}.csv', index=False)

    return f'/home/jun/workspace/KT/data/ednet/predict/{uid}.csv'

