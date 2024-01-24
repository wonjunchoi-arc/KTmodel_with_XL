import os
import sys
import pandas as pd
import numpy as np
import json
import copy
import tensorflow as tf


def read_data(fname : str, min_seq_len=3, response_set=[0, 1]) -> (pd.DataFrame, set):
    '''
    800,7 => userid, interaction len
    q4970,q6210,q5756,q5662,q307,q5421,q1315 => question
    74,121,74,87,30_24_48_181_182,78,38_39_181_185 => concepts
    1,0,0,0,1,0,0 => responses
    1562058133904,1562058172589,1562058207551,1562058239076,1562058278990,1562058298601,1562058319737 => timestamps
    25000,32000,30000,28000,37000,15000,18000 => usetimes

        Read the above txt file and make it a dataprame

    '''
    effective_keys = set()
    dres = dict()
    delstu, delnum, badr = 0, 0, 0
    goodnum = 0
    with open(fname, "r", encoding="utf8") as fin:
        i = 0
        lines = fin.readlines()
        dcur = dict()
        while i < len(lines):
            line = lines[i].strip()
            if i % 6 == 0:  # stuid
                effective_keys.add("uid")
                tmps = line.split(",")
                stuid, seq_len = tmps[0], int(tmps[1])
                if seq_len < min_seq_len:  # delete use seq len less than min_seq_len
                    i += 6
                    dcur = dict()
                    delstu += 1
                    delnum += seq_len
                    continue
                dcur["uid"] = stuid
                goodnum += seq_len
            elif i % 6 == 1:  # question ids / names
                qs = []
                if line.find("NA") == -1:
                    effective_keys.add("questions")
                    qs = line.split(",")
                dcur["questions"] = qs
            elif i % 6 == 2:  # concept ids / names
                cs = []
                if line.find("NA") == -1:
                    effective_keys.add("concepts")
                    cs = line.split(",")
                dcur["concepts"] = cs
            elif i % 6 == 3:  # responses
                effective_keys.add("responses")
                rs = []
                if line.find("NA") == -1:
                    flag = True
                    for r in line.split(","):
                        try:
                            r = int(r)
                            if r not in response_set:  # check if r in response set.
                                print(f"error response in line: {i}")
                                flag = False
                                break
                            rs.append(r)
                        except:
                            print(f"error response in line: {i}")
                            flag = False
                            break
                    if not flag:
                        i += 3
                        dcur = dict()
                        badr += 1
                        continue
                dcur["responses"] = rs
            elif i % 6 == 4:  # timestamps
                ts = []
                if line.find("NA") == -1:
                    effective_keys.add("timestamps")
                    ts = line.split(",")
                dcur["timestamps"] = ts
            elif i % 6 == 5:  # usets
                usets = []
                if line.find("NA") == -1:
                    effective_keys.add("usetimes")
                    usets = line.split(",")
                dcur["usetimes"] = usets

                for key in effective_keys:
                    dres.setdefault(key, [])
                    if key != "uid":
                        dres[key].append(",".join([str(k) for k in dcur[key]]))
                    else:
                        dres[key].append(dcur[key])
                dcur = dict()
            i += 1
    df = pd.DataFrame(dres)
    print(
        f"delete bad stu num of len: {delstu}, delete interactions: {delnum}, of r: {badr}, good num: {goodnum}")
    return df, effective_keys


def get_max_concepts(df: pd.DataFrame) -> int:

    max_concepts = 1
    for i, row in df.iterrows():
        cs = row["concepts"].split(",")
        num_concepts = max([len(c.split("_")) for c in cs])
        if num_concepts >= max_concepts:
            max_concepts = num_concepts
    return max_concepts



def calStatistics(df:pd.DataFrame, stares: list, key: str) ->(int,int,int,int,int):
    '''
    check and print data status
    '''
    allin, allselect = 0, 0
    allqs, allcs = set(), set()
    for i, row in df.iterrows():
        rs = row["responses"].split(",")
        curlen = len(rs) - rs.count("-1")
        allin += curlen
        if "selectmasks" in row:
            ss = row["selectmasks"].split(",")
            slen = ss.count("1")
            allselect += slen
        if "concepts" in row:
            cs = row["concepts"].split(",")
            fc = list()
            for c in cs:
                cc = c.split("_")
                fc.extend(cc)
            curcs = set(fc) - {"-1"}
            allcs |= curcs
        if "questions" in row:
            qs = row["questions"].split(",")
            curqs = set(qs) - {"-1"}
            allqs |= curqs
    stares.append(",".join([str(s)
                  for s in [key, allin, df.shape[0], allselect]]))
    return allin, allselect, len(allqs), len(allcs), df.shape[0]




def extend_multi_concepts(df : pd.DataFrame, effective_keys:set) -> (pd.DataFrame, set):
    '''
    When importing a tag according to the question from the file of kt1.content.csv, several concepts are included in one question as shown in 12_24.

    At this time, 12_24 is separated, 12,24 individual concepts are created, and the same responses are given
    '''
    if "questions" not in effective_keys or "concepts" not in effective_keys:
        print("has no questions or concepts! return original.")
        return df, effective_keys
    extend_keys = set(df.columns) - {"uid"}

    dres = {"uid": df["uid"]}
    for _, row in df.iterrows():
        dextend_infos = dict()
        for key in extend_keys:
            dextend_infos[key] = row[key].split(",")
        dextend_res = dict()
        for i in range(len(dextend_infos["questions"])):
            dextend_res.setdefault("is_repeat", [])
            if dextend_infos["concepts"][i].find("_") != -1:
                ids = dextend_infos["concepts"][i].split("_")
                dextend_res.setdefault("concepts", [])
                dextend_res["concepts"].extend(ids)
                for key in extend_keys:
                    if key != "concepts":
                        dextend_res.setdefault(key, [])
                        dextend_res[key].extend(
                            [dextend_infos[key][i]] * len(ids))
                dextend_res["is_repeat"].extend(
                    ["0"] + ["1"] * (len(ids) - 1))  # 1: repeat, 0: original
            else:
                for key in extend_keys:
                    dextend_res.setdefault(key, [])
                    dextend_res[key].append(dextend_infos[key][i])
                dextend_res["is_repeat"].append("0")
        for key in dextend_res:
            dres.setdefault(key, [])
            dres[key].append(",".join(dextend_res[key]))

    finaldf = pd.DataFrame(dres)
    effective_keys.add("is_repeat")
    return finaldf, effective_keys



def id_mapping(df:pd.DataFrame) ->(pd.DataFrame, dict):
    '''
    The question consisting of q123, q245, etc. is mapped from integer 0, The same goes for concepts
    '''
    id_keys = ["questions", "concepts", "uid"]
    dres = dict()
    dkeyid2idx = dict()
    print(f"df.columns: {df.columns}")
    for key in df.columns:
        if key not in id_keys:
            dres[key] = df[key]
    for i, row in df.iterrows():
        for key in id_keys:
            if key not in df.columns:
                continue
            dkeyid2idx.setdefault(key, dict()) # {'questions': {}, 'concepts': {}, 'uid': {}}
            dres.setdefault(key, []) # {'questions': [], 'concepts': [], 'uid': []}
            curids = []
            for id in row[key].split(","):
                if id not in dkeyid2idx[key]:
                    if key == "concepts":
                        dkeyid2idx[key][id] = len(dkeyid2idx[key])+4
                    elif key == "questions":
                        dkeyid2idx[key][id] = len(dkeyid2idx[key])+4
                    else :
                        dkeyid2idx[key][id] = len(dkeyid2idx[key])
                curids.append(str(dkeyid2idx[key][id]))
            dres[key].append(",".join(curids))
    finaldf = pd.DataFrame(dres)
    return finaldf, dkeyid2idx


def save_id2idx(dkeyid2idx, save_path):
    with open(save_path, "w+") as fout:
        fout.write(json.dumps(dkeyid2idx))

def train_test_split(df, test_ratio=0.2):
    df = df.sample(frac=1.0, random_state=1024)
    datanum = df.shape[0]
    test_num = int(datanum * test_ratio)
    train_num = datanum - test_num
    train_df = df[0:train_num]
    test_df = df[train_num:]
    # report
    print(
        f"total num: {datanum}, train+valid num: {train_num}, test num: {test_num}")
    return train_df, test_df

def save_dcur(row, effective_keys):
    dcur = dict()
    for key in effective_keys:
        if key not in ONE_KEYS:
            # [int(i) for i in row[key].split(",")]
            dcur[key] = row[key].split(",")
        else:
            dcur[key] = row[key]
    return dcur


def generate_sequences(df, effective_keys, min_seq_len=3, maxlen=200, pad_val=-1):
    save_keys = list(effective_keys) + ["selectmasks"]
    dres = {"selectmasks": []}
    dropnum = 0
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)

        rest, lenrs = len(dcur["responses"]), len(dcur["responses"])
        j = 0
        while lenrs >= j + maxlen:
            rest = rest - (maxlen)
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    # [str(k) for k in dcur[key][j: j + maxlen]]))
                    dres[key].append(",".join(dcur[key][j: j + maxlen]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * maxlen))

            j += maxlen
        if rest < min_seq_len:  # delete sequence len less than min_seq_len
            dropnum += rest
            continue

        pad_dim = maxlen - rest
        for key in effective_keys:
            dres.setdefault(key, [])
            if key not in ONE_KEYS:
                paded_info = np.concatenate(
                    [dcur[key][j:], np.array([pad_val] * pad_dim)])
                dres[key].append(",".join([str(k) for k in paded_info]))
            else:
                dres[key].append(dcur[key])
        dres["selectmasks"].append(
            ",".join(["1"] * rest + [str(pad_val)] * pad_dim))

    # after preprocess data, report
    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    print(f"dropnum: {dropnum}")
    return finaldf

def KFold_split(df, k=5):
    df = df.sample(frac=1.0, random_state=1024)
    datanum = df.shape[0]
    test_ratio = 1 / k
    test_num = int(datanum * test_ratio)
    rest = datanum % k

    start = 0
    folds = []
    for i in range(0, k):
        if rest > 0:
            end = start + test_num + 1
            rest -= 1
        else:
            end = start + test_num
        folds.extend([i] * (end - start))
        print(f"fold: {i+1}, start: {start}, end: {end}, total num: {datanum}")
        start = end
    # report
    finaldf = copy.deepcopy(df)
    finaldf["fold"] = folds
    return finaldf




def get_mask_tokens(R:list,mask_token:int , eos_token:int):
    '''
    For the input value, 15% is randomly changed to the mask value.
    R shows the input value with 15 masks, labels shows the original value of the part with the mask, and all parts without the mask are filled with -100.
    '''
    mlm_probability=0.15
    labels = tf.identity(R)
    
    probability_matrix = tf.fill(tf.shape(labels), mlm_probability)
    # special_token_mask = R > 1
    special_token_mask = tf.equal(R, eos_token)

    probability_matrix = tf.where(special_token_mask, 0.0, probability_matrix)
    masked_indices = tf.cast(tf.random.uniform(tf.shape(labels)) < probability_matrix, tf.bool)
    labels = tf.where(~masked_indices, -100, labels)

    indices_replaced = tf.logical_and(tf.random.uniform(tf.shape(labels)) < 0.8, masked_indices)
    R = tf.where(masked_indices,mask_token,R)
    # R = tf.where(indices_replaced,mask_token , R)

    # 이 아래로는 랜덤 단어 뽑아 내는 것
    # indices_random = tf.logical_and(
    #     tf.logical_and(tf.random.uniform(tf.shape(labels)) < 0.5, masked_indices),
    #     ~indices_replaced
    # )

    #random_words = tf.random.uniform(tf.shape(labels), minval=0, maxval=tokenizer.vocab_size, dtype=tf.int32)
    #input_ids = tf.where(indices_random, random_words, input_ids)

    return R, labels

def get_evalmask_token(R :list, mask_token : int, eos_token : int):
    '''
    Select a particular element with a 15% probability => 1, mask_a, 2, 3, 4, mask_b
    After randomly selecting one of the above particular element again, all the rear parts are masked based on the corresponding elements. =>  randomly select mask_a, mask_b and then all elements are masked
    behind mask_a or mask_b
    '''
    mlm_probability=0.15
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



def sta_infos(df, keys, stares, split_str="_"):
    # keys: 0: uid , 1: concept, 2: question
    uids = df[keys[0]].unique()
    if len(keys) == 2:
        cids = df[keys[1]].unique()
    elif len(keys) > 2:
        qids = df[keys[2]].unique()
        ctotal = 0
        cq = df.drop_duplicates([keys[2], keys[1]])[[keys[2], keys[1]]]
        cq[keys[1]] = cq[keys[1]].fillna("NANA")
        cids, dq2c = set(), dict()
        for i, row in cq.iterrows():
            q = row[keys[2]]
            ks = row[keys[1]]
            dq2c.setdefault(q, set())
            if ks == "NANA":
                continue
            for k in str(ks).split(split_str):
                dq2c[q].add(k)
                cids.add(k)
        ctotal, na, qtotal = 0, 0, 0
        for q in dq2c:
            if len(dq2c[q]) == 0:
                na += 1 # questions has no concept
                continue
            qtotal += 1
            ctotal += len(dq2c[q])
        
        avgcq = round(ctotal / qtotal, 4)
    avgins = round(df.shape[0] / len(uids), 4)
    ins, us, qs, cs = df.shape[0], len(uids), "NA", len(cids)
    avgcqf, naf = "NA", "NA"
    if len(keys) > 2:
        qs, avgcqf, naf = len(qids), avgcq, na
    curr = [ins, us, qs, cs, avgins, avgcqf, naf]
    stares.append(",".join([str(s) for s in curr]))
    return ins, us, qs, cs, avgins, avgcqf, naf

def write_txt(file, data):
    with open(file, "w") as f:
        for dd in data:
            for d in dd:
                f.write(",".join(d) + "\n")

from datetime import datetime
def change2timestamp(t, hasf=True):
    if hasf:
        timeStamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000
    else:
        timeStamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timestamp() * 1000
    return int(timeStamp)

def replace_text(text):
    text = text.replace("_", "####").replace(",", "@@@@")
    return text


def format_list2str(input_list):
    return [str(x) for x in input_list]


def one_row_concept_to_question(row):
    """Convert one row from concept to question

    Args:
        row (_type_): _description_

    Returns:
        _type_: _description_
    """
    new_question = []
    new_concept = []
    new_response = []

    tmp_concept = []
    begin = True
    for q, c, r, mask, is_repeat in zip(row['questions'].split(","),
                                        row['concepts'].split(","),
                                        row['responses'].split(","),
                                        row['selectmasks'].split(","),
                                        row['is_repeat'].split(","),
                                        ):
        if begin:
            is_repeat = "0"
            begin = False
        if mask == '-1':
            break
        if is_repeat == "0":
            if len(tmp_concept) != 0:
                new_concept.append("_".join(tmp_concept))
                tmp_concept = []
            new_question.append(q)
            new_response.append(r)
            tmp_concept = [c]
        else:#如果是 1 就累计知识点
            tmp_concept.append(c)
    if len(tmp_concept) != 0:
        new_concept.append("_".join(tmp_concept))

    if len(new_question) < 200:
        pads = ['-1'] * (200 - len(new_question))
        new_question += pads
        new_concept += pads
        new_response += pads

    new_selectmask = ['1']*len(new_question)
    new_is_repeat = ['0']*len(new_question)

    new_row = {"fold": row['fold'],
               "uid": row['uid'],
               "questions": ','.join(new_question),
               "concepts": ','.join(new_concept),
               "responses": ','.join(new_response),
               "selectmasks": ','.join(new_selectmask),
               "is_repeat": ','.join(new_is_repeat),
               }
    return new_row

def concept_to_question(df):
    """Convert df from concept to question
    Args:
        df (_type_): df contains concept

    Returns:
        _type_: df contains question
    """
    new_row_list = list(df.apply(one_row_concept_to_question,axis=1).values)
    df_new = pd.DataFrame(new_row_list)
    return df_new

def get_df_from_row(row):
    value_dict = {}
    for col in ['questions', 'concepts', 'responses', 'is_repeat']:
        value_dict[col] = row[col].split(",")
    df_value = pd.DataFrame(value_dict)
    df_value = df_value[df_value['questions']!='-1']
    return df_value