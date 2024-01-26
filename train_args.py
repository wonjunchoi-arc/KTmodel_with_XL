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



def make_tensorboard_summary_writer(config_xl):
    if not os.path.exists(config_xl.tensorboard_log_dir):
        os.makedirs(config_xl.tensorboard_log_dir)     
    train_log_dir = config_xl.tensorboard_log_dir+ current_time +'{}ep_{}mem_{}/train'.format(config_xl.epoch, config_xl.mem_len, config_xl.mode)
    test_log_dir = config_xl.tensorboard_log_dir+ current_time +'{}ep_{}mem_{}/test'.format(config_xl.epoch, config_xl.mem_len,config_xl.mode)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    logging.info('tensorboard_log_dir:  %s',config_xl.tensorboard_log_dir)

    return train_summary_writer, test_summary_writer



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
        loss = outputs.loss
        mems = outputs.mems
        loss_mx = target != -100
        loss_value = loss[loss_mx]
        loss_value = tf.reshape(loss_value, [-1, config_xl.R_vocab_size])
        labels = target[loss_mx]

        
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=loss_value)
        # batch_loss = tf.reduce_sum(loss) / valid_samples
        mean_loss = tf.reduce_mean(loss)
        train_loss(loss)
        train_accuracy(labels,loss_value)
        predictions =tf.nn.softmax(loss_value)
        train_auc(tf.one_hot(labels, depth=predictions.shape[1]), predictions)

    gradients = tape.gradient(mean_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return mems,mean_loss


def evaluate_step(model,test_dataset,test_summary_writer):
    total_loss = 0.0
    num_batches = 0
    evaluation_metrics = []
    test_mems = None

    for test_input_data, test_mask, test_labels in tqdm(test_dataset, desc='eval'):
    #     outputs = model(concepts=test_input_data, responses=test_mask, labels=test_labels, mems=mems, training=False)
    # for test_question,test_ceq, test_mask, test_labels in test_dataset:
    #     test_input_data = test_ceq if config_xl.mode == 'concepts' else test_question
        outputs = model(concepts=test_input_data, responses=test_mask, labels=test_labels, mems=test_mems, training=False)
        loss = outputs.loss
        test_mems = outputs.mems

        loss_mx = test_labels != -100
        loss_value = loss[loss_mx]
        loss_value = tf.reshape(loss_value, [-1, config_xl.R_vocab_size])
        labels = test_labels[loss_mx]

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

# make embedding projector 
def Make_embedding_projector(config_xl, dkeyid2idx,model):
    log_dir=config_xl.tensorboard_emb_log_dir+current_time+'_{}ep_{}mem_{}/'.format(config_xl.epoch, config_xl.mem_len, config_xl.mode)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save Labels separately on a line-by-line manner.
    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        for concepts in dkeyid2idx[config_xl.mode]:
            f.write("{}\n".format(concepts))

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



def train(config_xl):
    try:
        train_dataset,test_dataset,dkeyid2idx=load_TFdataset(config_xl)
        
        train_summary_writer, test_summary_writer = make_tensorboard_summary_writer(config_xl)

        learning_rate = CustomSchedule(config_xl.d_model)

        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        model = TFTransfoXLMLMHeadModel(config=config_xl)

        loss_values = []
        num_batches = 0

        for epoch in range(config_xl.epoch):
            start = time.time()
            total_loss = 0.0
            mems = None                   
            for input_data, mask, labels in tqdm(train_dataset, desc='train'):
                mems, loss_value = train_step(model, input_data,mask, labels, mems, optimizer)
            # for question,ceq, mask, labels in tqdm(train_dataset):
            #     input_data = ceq if config_xl.mode == 'concepts' else question
                num_batches += 1
                total_loss += loss_value.numpy()
                if num_batches % 100 == 0:
                    loss_values.append(loss_value.numpy())
                    print(f'Epoch {epoch + 1} Batch {num_batches} Loss {loss_value.numpy()}')
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), step=num_batches)
                    tf.summary.scalar('accuracy', train_accuracy.result(), step=num_batches)
                    tf.summary.scalar('auc', train_auc.result(), step=num_batches)

        # save model            
        if not os.path.exists(config_xl.model_save_dir):
            os.makedirs(config_xl.model_save_dir)
            model_saved_dir =config_xl.model_save_dir+current_time+'_{}ep_{}mem_{}.ckpt/my_checkpoint'.format(config_xl.epoch, config_xl.mem_len, config_xl.mode)       
            model.save_weights(model_saved_dir)

            logging.info('model.summary',model.summary()) 

    except Exception as e:
        logging.error(f"Error: {e}")

    return model,test_dataset,test_summary_writer
        
def evaluate(model,test_dataset,config_xl,test_summary_writer):
    try:
        

        model = TFTransfoXLMLMHeadModel(config=config_xl)

        test_loss0,test_acc0,average_precision, average_recall, average_f1_score = evaluate_step(model, test_dataset,test_summary_writer)
        logging.info('test_loss:{},test_acc:{},test_precision:{}, average_recall:{}, average_f1_score:{}'.format(test_loss0,test_acc0,average_precision, average_recall, average_f1_score))
    except Exception as e:
        logging.error(f"Error: {e}")



def main(config_xl) -> None :

    model,test_dataset,test_summary_writer =train(config_xl)
    evaluate(model,test_dataset, config_xl,test_summary_writer)

 
# nohup python /home/jun/workspace/KT/train.py 1 > 1.out 2 > 2.out &


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='TransfoXL config')
    parser.add_argument('--d_embed', type=int, required=False, default=128, help='Dimensionality of the embeddings')
    parser.add_argument('--d_head', type=int, required=False,default=32,help='Dimensionality of the model’s heads')
    parser.add_argument('--d_model', type=int, required=False,default=128 , help='Dimensionality of the model’s hidden states.')
    parser.add_argument('--d_inner', type=int, default=4096, help='Inner dimension in FF')
    parser.add_argument('--mask_token', type=int, required=False, default=3)
    parser.add_argument('--mem_len', type=int, required=True,default=600,help='Length of the retained previous heads')
    parser.add_argument('--n_head', type=int, required=False, default=8,help='Number of attention heads')
    parser.add_argument('--n_layer', type=int, required=False, default=6, help='Number of hidden layers in the Transformer encoder')
    parser.add_argument('--C_vocab_size', type=int, required=False, default=188,help='how many concepts')
    parser.add_argument('--Q_vocab_size', type=int, required=False, default=12277, help='how many questions')
    parser.add_argument('--R_vocab_size', type=int, required=False, default=2)
    parser.add_argument('--epoch', type=int, required=True, default=3)
    parser.add_argument('--mode', type=str, required=True, default='concepts',help='concepts or questions')
    parser.add_argument('--tf_data_dir', type=str, required=True, default='/home/jun/workspace/KT/data/ednet/TF_DATA')
    parser.add_argument('--tensorboard_log_dir', type=str, required=False, default='/home/jun/workspace/KT/logs/gradient_tape/')
    parser.add_argument('--tensorboard_emb_log_dir', type=str, required=False, default='/home/jun/workspace/KT/logs/embedding/',help='tensorboard embedding projection dictionary')
    parser.add_argument('--model_save_dir', type=str, required=False, default='/home/jun/workspace/KT/save_model/')
    args = parser.parse_args()


    config_xl = TransfoXLConfig(
        d_embed=args.d_embed,
        d_head = args.d_head,
        d_model=args.d_model,
        mem_len=args.mem_len,
        n_head=args.n_head,
        n_layer=args.n_layer,
        # batch_size = args.batch_size,
        # tgt_len = args.tgt_len,
        # eos_token=args.eos_token,
        mask_token=args.mask_token,
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


    main(config_xl)