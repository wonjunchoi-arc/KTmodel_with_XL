import pandas as pd
import random
import logging
import os
from data_utils import sta_infos, write_txt
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='make_txt')
parser.add_argument('--read_folder', type=str, required=True, default='/home/jun/workspace/KT/data/ednet/')
parser.add_argument('--name_txt', type=str, required=True)
parser.add_argument('--dataset_name', type=str, required=False, default='ednet')
args = parser.parse_args()


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 

formatter = logging.Formatter(fmt='%(asctime)s:%(module)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M')



# DEBUG 레벨 이상의 로그를 `debug.log`에 출력하는 Handler
file_debug_handler = logging.FileHandler('logs/make_txt/info.log')
file_debug_handler.setLevel(logging.INFO)
file_debug_handler.setFormatter(formatter)
logger.addHandler(file_debug_handler)

# ERROR 레벨 이상의 로그를 `error.log`에 출력하는 Handler
file_error_handler = logging.FileHandler('logs/make_txt/error.log')
file_error_handler.setLevel(logging.ERROR)
file_error_handler.setFormatter(formatter)
logger.addHandler(file_error_handler)



def read_data_from_csv(read_file : str, write_file :str,dataset_name :str) -> (str, str):
    KEYS = ["user_id", "tags", "question_id"]
    stares = []

    if not dataset_name is None:
        write_file = write_file.replace("/ednet/", f"/{dataset_name}/")
        write_dir = read_file.replace("/ednet/", f"/{dataset_name}")
        logger.info(f"write_dir is {write_dir}")
        logger.info(f"write_file is {write_file}")

    # uid range
    random.seed(2)
    samp = [i for i in range(840473)]
    random.shuffle(samp)


    #make individual uid.csv to one list
    count = 0
    file_list = list()
    for unum in tqdm(samp):
        str_unum = str(unum)
        df_path = os.path.join(read_file, f"KT1_sample/u{str_unum}.csv")
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            df['user_id'] = unum

            file_list.append(df)
            count = count + 1

        
    start_i =0
    logger.info(f"total user num: {count}")
    user_df = pd.concat(file_list[start_i:])
    logger.info(f"after sub user_df: {len(user_df)}")
    user_df["index"] = range(user_df.shape[0])
    question_df = pd.read_csv(os.path.join(read_file, 'contents', 'questions.csv'))
    
    if not dataset_name is None:
        read_file = write_dir 

    question_df['tags'] = question_df['tags'].apply(lambda x:x.replace(";","_"))
    question_df = question_df[question_df['tags']!='-1']
    merged_df = user_df.merge(question_df, sort=False,how='left')
    merged_df = merged_df.dropna(subset=["user_id", "question_id", "elapsed_time", "timestamp", "tags", "user_answer"])
    merged_df['correct'] = (merged_df['correct_answer']==merged_df['user_answer']).apply(int)


    ins, us, qs, cs, avgins, avgcq, na = sta_infos(merged_df, KEYS, stares)
    logger.info(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")


    ins, us, qs, cs, avgins, avgcq, na = sta_infos(merged_df, KEYS, stares)
    logger.info(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    
    ui_df = merged_df.groupby('user_id', sort=False)

    user_inters = []
    for ui in tqdm(ui_df):
        user, tmp_inter = ui[0], ui[1]
        tmp_inter = tmp_inter.sort_values(by=["timestamp", "index"])
        seq_len = len(tmp_inter)
        seq_skills = tmp_inter['tags'].astype(str)
        seq_ans = tmp_inter['correct'].astype(str)
        seq_problems = tmp_inter['question_id'].astype(str)
        seq_start_time = tmp_inter['timestamp'].astype(str)
        seq_response_cost = tmp_inter['elapsed_time'].astype(str)

        assert seq_len == len(seq_problems) == len(seq_ans)

        user_inters.append(
            [[str(user), str(seq_len)], seq_problems, seq_skills, seq_ans, seq_start_time, seq_response_cost])

    write_txt(write_file, user_inters)
    logger.info("\n".join(stares))
    return write_dir, write_file


# readf = '/home/jun/workspace/KT/data/ednet/'
readf=args.read_folder
dname = "/".join(args.read_folder.split("/")[0:-1])
writef = os.path.join(dname, args.name_txt)
dname, writef = read_data_from_csv(readf, writef, args.dataset_name)
print('dname',type(dname))
print('writef',type(writef))
