import os
import sys
import pandas as pd
import numpy as np
import json
import copy
import tensorflow as tf


def read_data(fname, min_seq_len=3, response_set=[0, 1]):
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


def get_max_concepts(df):
    max_concepts = 1
    for i, row in df.iterrows():
        cs = row["concepts"].split(",")
        num_concepts = max([len(c.split("_")) for c in cs])
        if num_concepts >= max_concepts:
            max_concepts = num_concepts
    return max_concepts



def calStatistics(df, stares, key):
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



def extend_multi_concepts(df, effective_keys):
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



def id_mapping(df):
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




def get_mask_tokens(R,mask_token, mlm_probability=0.15, special_token_cnt=3):
    labels = tf.identity(R)
    
    probability_matrix = tf.fill(tf.shape(labels), mlm_probability)
    # special_token_mask = R > 1
    special_token_mask = tf.equal(R, 2)

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