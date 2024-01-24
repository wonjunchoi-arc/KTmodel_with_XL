Useage

This pipeline loads the Raw KT data U{}.csv, content/question.csv

Create a column such as user_id, timestamp, question_id, concepts, responses, and elapsed_time and save it as a txt file.

After that, load the txt file and create a TF.DATASET to use as an input for the model, and train, test, and report.

1. Install the requirements
2. Download/Prepare the dataset
3. Train and test transfor_xl mlm model
4. Check the train, test report using tensorboard

**Readme**

MAKE TXT FilE

KT1 Data(u{}.csv, content/question.csv) -> data.txt

u{}.csv,

|  | user_id | timestamp | solving_id | v | user_answer | elapsed_time |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | u717875 | 1565332027449 | 1 | q4862 | d | 45000 |
| 1 | u717875 | 1565332057492 | 2 | q6747 | d | 24000 |
| 2 | u717875 | 1565332085743 | 3 | q326 | c | 25000 |

content/question.csv

|  | question_id | bundle_id | explanation_id | correct_answer | part | tags | deployed_at |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | q2 | b2 | e2 | a | 1 | 15;2;182 | 1558093219720 |
| 1 | q3 | b3 | e3 | b | 1 | 14;2;179;183 | 1558093222784 |
|  |  |  |  |  |  |  |  |

약 78만개의 u{}.csv 파일을 모두 합치고 responses, concepts column을 추가한뒤 txt로 변환 및 저장

responses : 4지 선다형 응답 세트를 question.csv 파일의 정답 세트와 비교하여 맞으면 1, 틀리면 0 으로

concepts: question.csv 파일의 question_id별 tag값

Python make_txt.py —read_folder {Path to the parent folder containing KT1/u{}.csv, content/quest.csv}  —name_txt {txt file name you want to save}

Train.py

CUDA_VISIBLE_DEVICES=0 python /home/jun/workspace/KT/train.py --mem_len 200 --batch_size 65 --tgt_len 140 --epoch 3 --mode concepts

MAKE TF.DATASET

아래와 같이 만들어진 txt 파일을 읽어와서 TF.DATASET으로 변환 및 저장

800,7
q4970,q6210,q5756,q5662,q307,q5421,q1315
74,121,74,87,30_24_48_181_182,78,38_39_181_185
1,0,0,0,1,0,0
1562058133904,1562058172589,1562058207551,1562058239076,1562058278990,1562058298601,1562058319737
25000,32000,30000,28000,37000,15000,18000

mode에 따라 Question, Concepts 데이터가  만들어진다.
