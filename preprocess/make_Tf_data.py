
import tensorflow as tf
from data_utils import read_data,get_max_concepts,calStatistics,extend_multi_concepts,id_mapping,save_id2idx,train_test_split,save_dcur,get_evalmask_token, get_mask_tokens
import itertools
import pickle
import os
import argparse

from tqdm import tqdm

def make_dataset(args):

    print('reading txt data ~~~~~~~')
    total_df, effective_keys = read_data(args.data)

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
    print('extend_multi_concepts ~ ')
    total_df_ex, effective_keys = extend_multi_concepts(total_df, effective_keys)
    print('in the process of mapping')
    total_df, dkeyid2idx = id_mapping(total_df_ex)
    dkeyid2idx["max_concepts"] = max_concepts

    extends, _, qs, cs, seqnum = calStatistics(
        total_df, stares, "extend multi")
    print("="*20)
    print(
        f"after extend multi, total interactions: {extends}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")


    #train test 분리
    train_df, test_df = train_test_split(total_df, args.test_ratio)

    #transformer_xl dataset 만들기
    train = {"qseqs": [], "cseqs": [], "masked_R": [], "labels": []} #"q_shift": [], "c_shift": [], "r_shift": []}
    test = {"qseqs": [], "cseqs": [], "masked_R": [], "labels": []} #"q_shift": [], "c_shift": [], "r_shift": []}


    # Make the value '12,52,1' to 12,52,1 and attach eos_token to concepts and questions by uid.
    for i, row in tqdm(train_df.iterrows(),desc="Train_df"):
        cseq_list=[int(_) for _ in row["concepts"].split(",")]
        cseq_list.append(args.eos_token)
        train["cseqs"].append(cseq_list)

        qes_list=[int(_) for _ in row["questions"].split(",")]
        qes_list.append(args.eos_token)
        train["qseqs"].append(qes_list)


        rseq_list=[(int(_)) for _ in row["responses"].split(",")]
        rseq_list.append(args.eos_token)
        masked_R, labels = get_mask_tokens(rseq_list,args.mask_token, args.eos_token)
        train["masked_R"].append(masked_R)
        train["labels"].append(labels)


    for i, row in tqdm(test_df.iterrows(),desc="Test_df"):
        cseq_list=[int(_) for _ in row["concepts"].split(",")]
        cseq_list.append(args.eos_token)   
        test["cseqs"].append(cseq_list)

        qes_list=[int(_) for _ in row["questions"].split(",")]
        qes_list.append(args.eos_token)
        test["qseqs"].append(qes_list)


        rseq_list=[(int(_)) for _ in row["responses"].split(",")]
        rseq_list.append(args.eos_token)
        masked_R, labels = get_evalmask_token(rseq_list, args.mask_token, args.eos_token)
        
        test["masked_R"].append(masked_R)
        test["labels"].append(labels)\
        



        #train data  
        #Work out how cleanly we can divide the dataset into bsz,tgt_len parts
        cseqs_list = list(itertools.chain(*train['cseqs']))
        qseqs_list = list(itertools.chain(*train['qseqs']))
        r_masked_list = tf.concat(train['masked_R'], axis=0)
        labels = tf.concat(train['labels'], axis=0)

        n_step = len(cseqs_list) // (args.batch_size*args.tgt_len)


        sliced_cseqs = tf.slice(cseqs_list,[0],[n_step * args.batch_size*args.tgt_len])  
        sliced_qseqs = tf.slice(qseqs_list,[0],[n_step * args.batch_size*args.tgt_len])  
        sliced_r_mask = tf.slice(r_masked_list,[0],[n_step * args.batch_size*args.tgt_len]) 
        sliced_labels = tf.slice(labels,[0],[n_step * args.batch_size*args.tgt_len]) 

        count =len(sliced_cseqs)// (args.batch_size*args.tgt_len)

        new_shape = (args.batch_size, -1)  # 나머지 차원은 자동으로 계산됨

        cseq_reshaped = tf.reshape(sliced_cseqs, new_shape)
        qseq_reshaped = tf.reshape(sliced_qseqs, new_shape)
        r_mask_reshaped = tf.reshape(sliced_r_mask, new_shape)
        labels_reshaped = tf.reshape(sliced_labels, new_shape)

        # Because of the einsum calculation of the xl model dataset batch, tgt_len dimension changed
        cseq_transposed = tf.transpose(cseq_reshaped)
        qseq_transposed = tf.transpose(qseq_reshaped)
        r_mask_transposed = tf.transpose(r_mask_reshaped)
        labels_transposed = tf.transpose(labels_reshaped)


        #test data  
        #Work out how cleanly we can divide the dataset into bsz,tgt_len parts        
        test_cseqs_list = list(itertools.chain(*test['cseqs']))
        test_qseqs_list = list(itertools.chain(*test['qseqs']))
        test_r_masked_list = tf.concat(test['masked_R'], axis=0)
        test_labels = tf.concat(test['labels'], axis=0)



        # Work out how cleanly we can divide the dataset into bsz parts.
        # 아래의 두 코드는   data 텐서에서 배치 크기 bsz로 깔끔하게 맞지 않는 추가 요소를 제거하는 것 배치에 띡 떨어지게
            
        test_n_step = len(test_cseqs_list) // (args.batch_size*args.tgt_len)
        print('n_step',test_n_step) # 

        test_sliced_cseqs = tf.slice(test_cseqs_list,[0],[test_n_step * args.batch_size*args.tgt_len])  
        test_sliced_qseqs = tf.slice(test_qseqs_list,[0],[test_n_step * args.batch_size*args.tgt_len])  
        test_sliced_r_masked_seq = tf.slice(test_r_masked_list,[0],[test_n_step * args.batch_size*args.tgt_len]) 

        test_sliced_labels = tf.slice(test_labels,[0],[test_n_step * args.batch_size*args.tgt_len])  

        count =len(test_sliced_cseqs)// (args.batch_size*args.tgt_len)
        print(count)



        new_shape = (args.batch_size, -1)  # 나머지 차원은 자동으로 계산됨

        test_qseq_reshaped = tf.reshape(test_sliced_qseqs, new_shape)
        test_cseq_reshaped = tf.reshape(test_sliced_cseqs, new_shape)
        test_r_masked_seq_reshaped = tf.reshape(test_sliced_r_masked_seq, new_shape)
        test_labels_reshaped = tf.reshape(test_sliced_labels, new_shape)


        test_qseq_transposed = tf.transpose(test_qseq_reshaped)
        test_cseq_transposed = tf.transpose(test_cseq_reshaped)
        test_r_masked_transposed = tf.transpose(test_r_masked_seq_reshaped)
        test_labels_transposed = tf.cast(tf.transpose(test_labels_reshaped),tf.int64)




        #make tf.dataset
        if args.mode =='question':
            train_dataset = tf.data.Dataset.from_tensor_slices(
            (qseq_transposed, r_mask_transposed, labels_transposed))
            test_dataset = tf.data.Dataset.from_tensor_slices(
            (test_qseq_transposed, test_r_masked_transposed, test_labels_transposed))
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices(
            (cseq_transposed, r_mask_transposed, labels_transposed))
            test_dataset = tf.data.Dataset.from_tensor_slices(
            (test_cseq_transposed, test_r_masked_transposed, test_labels_transposed))



        train_dataset =train_dataset.batch(args.tgt_len)
        test_dataset =test_dataset.batch(args.tgt_len)

        
        tf.data.experimental.save(train_dataset, args.tf_data_dir+'/{}'.format(args.mode)+'/train')
        tf.data.experimental.save(test_dataset, args.tf_data_dir+'/{}'.format(args.mode)+'/test')
        with open(args.tf_data_dir+"dkeyid2idx.pkl", "wb") as file:
            pickle.dump(dkeyid2idx, file)
            
        print('Making Tf.dataset is complited')
    

parser = argparse.ArgumentParser(description='TransfoXL config')
parser.add_argument('--data', type=str, required=True, default='/home/jun/workspace/KT/data/ednet/all_data.txt', help='Put the path to the txt file created through make_txt.py')
parser.add_argument('--batch_size', type=int, required=True, default=65)
parser.add_argument('--tgt_len', type=int, required=True, default=140)
parser.add_argument('--eos_token', type=int, required=False, default=2, help='End of stream token id.')
parser.add_argument('--mask_token', type=int, required=False, default=3)
parser.add_argument('--test_ratio', type=float, required=False, default=0.2)
parser.add_argument('--tf_data_dir', type=str, required=False, default='/home/jun/workspace/KT/data/ednet/TF_DATA')
parser.add_argument('--mode', type=str, required=True, default='concepts', help="concepts or questions")

args = parser.parse_args()


if not (os.path.exists(args.tf_data_dir)) & (os.path.exists(args.tf_data_dir+'/'+args.mode)):
    os.makedirs(args.tf_data_dir) 
    tf_train_dir = args.tf_data_dir+'/{}'.format(args.mode) +'/train'
    tf_test_dir = args.tf_data_dir+'/{}'.format(args.mode) +'/test'
    make_dataset(args)