import os
import pickle
import tensorflow as tf
import time
from models.model_for_kt import TFTransfoXLModel,TFTransfoXLLMHeadModel,TFTransfoXLMLMHeadModel
from transformers import TransfoXLConfig
from tensorflow.keras.utils import register_keras_serializable
from tqdm import tqdm
import datetime
import argparse
import logging
from tensorboard.plugins import projector
import numpy as np

import mlflow
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec


parser = argparse.ArgumentParser(description='TransfoXL config')
parser.add_argument('--d_embed', type=int, required=False, default=128, help='Dimensionality of the embeddings')
parser.add_argument('--d_head', type=int, required=False,default=32,help='Dimensionality of the model’s heads')
parser.add_argument('--d_model', type=int, required=False,default=128 , help='Dimensionality of the model’s hidden states.')
parser.add_argument('--d_inner', type=int, default=4096, help='Inner dimension in FF')
parser.add_argument('--mask_token', type=int, required=False, default=3)
parser.add_argument('--eos_token', type=int, required=False, default=2)
parser.add_argument('--batch_size', type=int, required=False, default=65)
parser.add_argument('--tgt_len', type=int, required=False, default=140)
parser.add_argument('--mem_len', type=int, required=False,default=600,help='Length of the retained previous heads')
parser.add_argument('--n_head', type=int, required=False, default=8,help='Number of attention heads')
parser.add_argument('--n_layer', type=int, required=False, default=6, help='Number of hidden layers in the Transformer encoder')
parser.add_argument('--C_vocab_size', type=int, required=False, default=188,help='how many concepts')
parser.add_argument('--Q_vocab_size', type=int, required=False, default=12277, help='how many questions')
parser.add_argument('--R_vocab_size', type=int, required=False, default=2)
parser.add_argument('--epoch', type=int, required=False, default=1)
parser.add_argument('--mode', type=str, required=True, default='concepts',help='concepts or questions')
parser.add_argument('--tf_data_dir', type=str, required=False, default='/home/jun/workspace/KT/data/ednet/100_sam')
parser.add_argument('--devices', type=str, required=True, default='gpu')
parser.add_argument('--tensorboard_log_dir', type=str, required=False, default='/home/jun/workspace/KT/logs/gradient_tape')
parser.add_argument('--tensorboard_emb_log_dir', type=str, required=False, default='/home/jun/workspace/KT/logs/embedding',help='tensorboard embedding projection dictionary')
parser.add_argument('--model_save_dir', type=str, required=False, default='/home/jun/workspace/KT/save_model')
args = parser.parse_args()

if args.devices == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# Set up train eval Metric
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


# Set up logging configuration
logging.basicConfig(level=logging.INFO)

def load_TFdataset(config_xl) :
    tf_train_dir = config_xl.tf_data_dir+'/{}'.format(config_xl.mode)+'/train'
    tf_test_dir = config_xl.tf_data_dir+'/{}'.format(config_xl.mode)+'/test'
    train_dataset = tf.data.experimental.load(tf_train_dir)
    test_dataset = tf.data.experimental.load(tf_test_dir)
    with open(config_xl.tf_data_dir+"/dkeyid2idx.pkl", "rb") as file:
        dkeyid2idx = pickle.load(file) 
    
    return train_dataset,test_dataset,dkeyid2idx





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
    
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    def get_config(self):
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
            }


@tf.function
def train_step(model, data1,data2, target, mems, optimizer):
    with tf.GradientTape() as tape:
        outputs = model(concepts=data1,responses=data2, labels=target, mems=mems)
        logit = outputs.logit
        mems = outputs.mems
        logit_mx = target != -100
        logit_value = logit[logit_mx]
        logit_value = tf.reshape(logit_value, [-1, config_xl.R_vocab_size])
        labels = target[logit_mx]

        
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logit_value)
        # batch_loss = tf.reduce_sum(loss) / valid_samples
        mean_loss = tf.reduce_mean(loss)
        train_loss(loss)
        train_accuracy(labels,logit_value)
        predictions =tf.nn.softmax(logit_value)
        train_auc(tf.one_hot(labels, depth=predictions.shape[1]), predictions)

    gradients = tape.gradient(mean_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return mems,mean_loss


def evaluate(model,test_dataset,config_xl):
    total_loss = 0.0
    num_batches = 0
    evaluation_metrics = []
    test_mems = None

    for input_data, masked_responses, responses in tqdm(test_dataset, desc='eval'):

        outputs = model(concepts=input_data, responses=masked_responses, labels=responses, mems=test_mems, training=False)
        logit = outputs.logit
        test_mems = outputs.mems

        logit_mx = responses != -100
        logit_value = logit[logit_mx]
        logit_value = tf.reshape(logit_value, [-1, config_xl.R_vocab_size])
        labels = responses[logit_mx]

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logit_value)
        mean_loss = tf.reduce_mean(loss)

        # Update precision and recall metrics
        predicted_labels = tf.argmax(logit_value, axis=1)
        predictions =tf.nn.softmax(logit_value)

        
        test_auc(tf.one_hot(labels, depth=predictions.shape[1]), predictions)
        test_precision(labels, predicted_labels)
        test_recall(labels, predicted_labels)

        test_accuracy(labels, logit_value)
        test_loss(loss)
        
        
        precision = test_precision.result().numpy()
        recall = test_recall.result().numpy()
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

        evaluation_metrics.append(test_accuracy.result().numpy())

        total_loss += mean_loss.numpy()
        num_batches += 1

        mlflow.log_metric('test_loss', test_loss.result(), step=num_batches)
        mlflow.log_metric('test_accuracy', test_accuracy.result(), step=num_batches)
        mlflow.log_metric('test_precision', test_precision.result(), step=num_batches)
        mlflow.log_metric('test_recall', test_recall.result(), step=num_batches)
        mlflow.log_metric('test_f1_score', f1_score, step=num_batches)
        mlflow.log_metric('test_auc', test_auc.result(), step=num_batches)

    # 평균 정밀도, 재현율, F1 점수를 계산합니다.
    average_precision = test_precision.result().numpy()
    average_recall = test_recall.result().numpy()
    average_f1_score = 2 * (average_precision * average_recall) / (average_precision + average_recall + 1e-7)

    average_metric = sum(evaluation_metrics) / len(evaluation_metrics)
    average_loss = total_loss / num_batches

    return average_loss, average_metric, average_precision, average_recall, average_f1_score

# make embedding projector 
def Make_embedding_projector(model,config_xl, dkeyid2idx,):
    log_dir=config_xl.tensorboard_emb_log_dir+'/'+current_time+'_{}ep_{}mem_{}/'.format(config_xl.epoch, config_xl.mem_len, config_xl.mode)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save Labels separately on a line-by-line manner.
    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        for valeu_before_mapping in dkeyid2idx[config_xl.mode]:
            f.write("{}\n".format(valeu_before_mapping))

    weights = tf.Variable(model.transformer.word_emb_C.get_weights()[0])

    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))
    # Set up config.
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)



def train(train_dataset,config_xl):
    try:
        learning_rate = CustomSchedule(config_xl.d_model)

        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        model = TFTransfoXLMLMHeadModel(config=config_xl)

        loss_values = []
        num_batches = 0

        for epoch in range(config_xl.epoch):
            start = time.time()
            total_loss = 0.0
            mems = None                   
            for input_data, masked_responses, responses in tqdm(train_dataset, desc='train'):
                mems, loss_value = train_step(model, input_data,masked_responses, responses, mems, optimizer)
                num_batches += 1
                total_loss += loss_value.numpy()
                # if num_batches % 100 == 0:
                loss_values.append(loss_value.numpy())
                print(f'Epoch {epoch + 1} Batch {num_batches} Loss {loss_value.numpy()}')
                mlflow.log_metric('loss', train_loss.result(), step=num_batches)
                mlflow.log_metric('accuracy', train_accuracy.result(), step=num_batches)
                mlflow.log_metric('auc', train_auc.result(), step=num_batches)


    except Exception as e:
        logging.error(f"Error: {e}")

    return model
        


def main(config_xl) -> None :
    train_dataset,test_dataset,dkeyid2idx=load_TFdataset(config_xl)
    model =train(train_dataset.take(10),config_xl)
    test_loss,test_acc,test_precision, test_recall, test_f1_score = evaluate(model, test_dataset,config_xl)
    # Make_embedding_projector(model,config_xl,dkeyid2idx)
    logging.info('test_loss:{},test_acc:{},test_precision:{}, test_recall:{}, test_f1_score:{}'.format(test_loss,test_acc,test_precision, test_recall, test_f1_score))
    
    # Infer the model signature
    input_data, masked_responses, responses = next(iter(test_dataset))
    outputs = model(concepts=input_data, responses=masked_responses, labels=responses, mems=None, training=False)
    logit = outputs.logit
    logit_value = tf.reshape(logit, [-1, config_xl.R_vocab_size])
    predicted_labels = tf.argmax(logit_value, axis=1)

    # transposed_response = tf.transpose(masked_responses)
    # 모델 입력과 출력에 대한 TensorSpec 정의
    input_schema = Schema(
    [
        TensorSpec(np.dtype(np.int32), (-1,len(input_data[1].numpy())), "input_data"),
        TensorSpec(np.dtype(np.int32), (-1,len(responses[1].numpy())), "responses"),
    ]
)
    output_schema = Schema([TensorSpec(np.dtype(np.int32),predicted_labels.numpy().shape, 'predicted_labels')])


    signature = ModelSignature(input_schema)


    # Log the model
    model_info = mlflow.tensorflow.log_model(
        model=model,
        artifact_path="iris_model",
        signature=signature,
        # input_example={"first_input": input_data, "second_input": masked_responses},
        # input_example=[input_data,masked_responses],
        registered_model_name="tracking-quickstart",
    )
    logging.info('model_info.model_uri: %s', model_info.model_uri)
    # mlflow.tensorflow.log_model(model, "model", signature=signature,registered_model_name="tracking-quickstart")

if __name__ == "__main__":

    config_xl = TransfoXLConfig(
            d_embed=args.d_embed,
            d_head = args.d_head,
            d_model=args.d_model,
            mem_len=args.mem_len,
            n_head=args.n_head,
            n_layer=args.n_layer,
            eos_token = args.eos_token,
            mask_token=args.mask_token,
            batch_size=args.batch_size,
            tgt_len=args.tgt_len,
            C_vocab_size=args.C_vocab_size,
            Q_vocab_size = args.Q_vocab_size,
            R_vocab_size = args.R_vocab_size,
            epoch = args.epoch,
            mode = args.mode, # concepts or questions 
            tf_data_dir = args.tf_data_dir,
            tensorboard_log_dir = args.tensorboard_log_dir,
            tensorboard_emb_log_dir = args.tensorboard_emb_log_dir,
            model_save_dir = args.model_save_dir
        )
 
    logging.info('config_xl:  %s',config_xl)
    # Set our tracking server uri for logging
    # mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    # Create a new MLflow Experiment
    mlflow.set_experiment("MLflow Test")

    # Start an MLflow run
    with mlflow.start_run():
        #set a run name
        mlflow.set_tag("mlflow.runName", '{}ep_{}mem_{}'.format(args.epoch,args.mem_len, args.mode))
        
        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", '{}ep_{}mem_{}'.format(args.epoch,args.mem_len, args.mode))

        # Log the hyperparameters
        mlflow.log_params(config_xl.to_dict())
        # mlflow.tensorflow.autolog()


        main(config_xl)