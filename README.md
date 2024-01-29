## Useage

This pipeline loads the Raw KT data U{}.csv, content/question.csv

Create a column such as user_id, timestamp, question_id, concepts, responses, and elapsed_time and save it as a txt file.

After that, load the txt file and create a TF.DATASET to use as an input for the model, and train, test, and report.

'''
1. Install the requirements
2. Download/Prepare the dataset
3. Train and test transfor_xl mlm model
4. Check the train, test report using tensorboard
'''




## make_txt.py

약 78만개의 u{}.csv 파일을 모두 합치고 responses, concepts column을 추가한뒤 txt로 변환 및 저장

responses : 4지 선다형 응답 세트를 question.csv 파일의 정답 세트와 비교하여 맞으면 1, 틀리면 0 으로

concepts: question.csv 파일의 question_id별 tag값

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



KT1 Data(u{}.csv, content/question.csv) -> data.txt

>  Python preprocess/make_txt.py —read_folder {Path to the parent folder containing KT1/u{}.csv, content/quest.csv}  —name_txt {txt file name you want to save}





## make_Tf_data.py

Load the txt file and create individual Tf.dataset (train, test) based on concepts, questions mode.

In addition, concepts (12_42) made of train, test split/tag, etc. are separated into individual elements and even mapping functions.
And  Reshape, transpose data  to enter the xl model, The row is batch_size and the column is tgt_len
Finally, save the tf.data.Dataset made like this


 > python preprocess/make_Tf_data.py --data {Path to txt.py} --batch_size --tgt_len --mode {concepts or questions} --tf_data_dir {Where you want to save it}





## train_args.py

> CUDA_VISIBLE_DEVICES=0 python train_args.py --tf_data_dir {path to your Tf.dataset dir} --mem_len 200 --batch_size 65 --tgt_len 140 --epoch 3 --mode {If you created concepts dataset, enter concepts mode}



## Make Docker images

> docker build -t custom/mydocker:latest --build-arg UID=93 --build-arg USER_NAME=jun -f Dockerfile .


## Start Docker Container

> docker run -it --name test --gpus 0 -p 8888:8888 -v $PWD:/workspace {Mount the folder running the current command and the docker's /workspace}  
> -w /workspace {Setting the Docker's work Folder} custom/mydocker:latest {docker images} /bin/bash


## Run Fast-api server

> uvicorn app.main:app --reload --host 0.0.0.0 --port 8888