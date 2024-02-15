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


# Set up logging configuration
logging.basicConfig(level=logging.INFO)


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


class TransformerXLTrainer:
    def __init__(self, args):
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')
        self.test_precision = tf.metrics.Precision()
        self.test_recall = tf.metrics.Recall()
        self.train_auc = tf.keras.metrics.AUC()
        self.test_auc = tf.keras.metrics.AUC()

        self.args = args
        self.config_xl = TransfoXLConfig(
            d_embed=args.d_embed,
            d_head=args.d_head,
            d_model=args.d_model,
            mem_len=args.mem_len,
            n_head=args.n_head,
            n_layer=args.n_layer,
            eos_token=args.eos_token,
            mask_token=args.mask_token,
            batch_size=args.batch_size,
            tgt_len=args.tgt_len,
            C_vocab_size=args.C_vocab_size,
            Q_vocab_size=args.Q_vocab_size,
            R_vocab_size=args.R_vocab_size,
            epoch=args.epoch,
            mode=args.mode,
            tf_data_dir=args.tf_data_dir,
            tensorboard_emb_log_dir=args.tensorboard_emb_log_dir,
            tensorboard_log_dir = args.tensorboard_log_dir,
            model_save_dir = args.model_save_dir
            # mlflow_tracking_uri=args.mlflow_tracking_uri
        )
        self.learning_rate = CustomSchedule(self.config_xl.d_model)


    
    def load_TFdataset(self) ->(tf.data.Dataset, tf.data.Dataset, dict) :
        tf_train_dir = self.config_xl.tf_data_dir+'/{}'.format(self.config_xl.mode)+'/train'
        tf_test_dir = self.config_xl.tf_data_dir+'/{}'.format(self.config_xl.mode)+'/test'
        train_dataset = tf.data.experimental.load(tf_train_dir)
        test_dataset = tf.data.experimental.load(tf_test_dir)
        with open(self.config_xl.tf_data_dir+"/dkeyid2idx.pkl", "rb") as file:
            dkeyid2idx = pickle.load(file) 
        return train_dataset,test_dataset,dkeyid2idx



    def make_tensorboard_summary_writer(self):
        if not os.path.exists(self.config_xl.tensorboard_log_dir):
            os.makedirs(self.config_xl.tensorboard_log_dir)     
        train_log_dir = self.config_xl.tensorboard_log_dir+'/'+ current_time +'{}ep_{}mem_{}/train'.format(self.config_xl.epoch, self.config_xl.mem_len, self.config_xl.mode)
        test_log_dir = self.config_xl.tensorboard_log_dir+'/'+ current_time +'{}ep_{}mem_{}/test'.format(self.config_xl.epoch, self.config_xl.mem_len,self.config_xl.mode)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        logging.info('tensorboard_log_dir:  %s',self.config_xl.tensorboard_log_dir)

        return train_summary_writer, test_summary_writer


    @tf.function
    def train_step(self, model, data1,data2, target, mems, optimizer) ->  (list, tf.Tensor):
        with tf.GradientTape() as tape:
            outputs = model(concepts=data1,responses=data2, labels=target, mems=mems)
            logit = outputs.logit
            mems = outputs.mems
            logit_mx = target != -100
            logit_value = logit[logit_mx]
            logit_value = tf.reshape(logit_value, [-1, self.config_xl.R_vocab_size])
            labels = target[logit_mx]

            
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logit_value)
            # batch_loss = tf.reduce_sum(loss) / valid_samples
            mean_loss = tf.reduce_mean(loss)
            self.train_loss(loss)
            self.train_accuracy(labels,logit_value)
            predictions =tf.nn.softmax(logit_value)
            self.train_auc(tf.one_hot(labels, depth=predictions.shape[1]), predictions)

        gradients = tape.gradient(mean_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return mems,mean_loss


    def evaluate(self,model,test_dataset,test_summary_writer):
        total_loss = 0.0
        num_batches = 0
        test_mems = None

        for input_data, masked_responses, responses in tqdm(test_dataset.take(2), desc='eval'):
            outputs = model(concepts=input_data, responses=masked_responses, labels=responses, mems=test_mems, training=False)
            logit = outputs.logit
            test_mems = outputs.mems

            logit_mx = responses != -100
            logit_value = logit[logit_mx]
            logit_value = tf.reshape(logit_value, [-1, self.config_xl.R_vocab_size])
            labels = responses[logit_mx]

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logit_value)
            mean_loss = tf.reduce_mean(loss)

            # Update precision and recall metrics
            predicted_labels = tf.argmax(logit_value, axis=1)
            predictions =tf.nn.softmax(logit_value)

            
            self.test_auc(tf.one_hot(labels, depth=predictions.shape[1]), predictions)
            self.test_precision(labels, predicted_labels)
            self.test_recall(labels, predicted_labels)
            self.test_accuracy(labels, logit_value)
            
            
            # precision = test_precision.result().numpy()
            # recall = test_recall.result().numpy()
            # f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

            # evaluation_metrics.append(test_accuracy.result().numpy())

            total_loss += mean_loss.numpy()
            num_batches += 1

            
            

        # 평균 정밀도, 재현율, F1 점수를 계산합니다.
        average_accuracy =self.test_accuracy.result().numpy()
        average_precision = self.test_precision.result().numpy()
        average_recall = self.test_recall.result().numpy()
        average_f1_score = 2 * (average_precision * average_recall) / (average_precision + average_recall + 1e-7)
        average_auc=self.test_auc.result().numpy()
        
        with test_summary_writer.as_default():
            tf.summary.scalar('accuracy', self.test_accuracy.result(), step=num_batches)
            tf.summary.scalar('precision', self.test_precision.result(), step=num_batches)
            tf.summary.scalar('recall', self.test_recall.result(), step=num_batches)
            tf.summary.scalar('f1_score', average_f1_score, step=num_batches)
            tf.summary.scalar('auc', self.test_auc.result(), step=num_batches)
        
        logging.info('test_acc:{}, average_f1_score:{}, average_auc:{}'.format(average_accuracy, average_f1_score,average_auc))

        # return average_accuracy, average_precision, average_recall, average_f1_score,average_auc

    # make embedding projector 
    def Make_embedding_projector(self,model, dkeyid2idx):
        log_dir=self.config_xl.tensorboard_emb_log_dir+'/'+current_time+'_{}ep_{}mem_{}/'.format(self.config_xl.epoch, self.config_xl.mem_len, self.config_xl.mode)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Save Labels separately on a line-by-line manner.
        with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
            for valeu_before_mapping in dkeyid2idx[self.config_xl.mode]:
                f.write("{}\n".format(valeu_before_mapping))

        weights = tf.Variable(model.transformer.word_emb_C.get_weights()[0])

        checkpoint = tf.train.Checkpoint(embedding=weights)
        checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))
        # Set up self.config_xl.
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(log_dir,config)



    def train_test(self,train_dataset,test_dataset,train_summary_writer,test_summary_writer):
        try:
            
            optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

            model = TFTransfoXLMLMHeadModel(config= self.config_xl)

            loss_values = []
            num_batches = 0

            for epoch in range(self.config_xl.epoch):
                start = time.time()
                total_loss = 0.0
                mems = None                   
                for input_data, masked_responses, responses in tqdm(train_dataset.take(2), desc='train'):
                    mems, loss_value = self.train_step(model, input_data,masked_responses, responses, mems, optimizer)
                    num_batches += 1
                    total_loss += loss_value.numpy()
                    if num_batches % 100 == 0:
                        loss_values.append(loss_value.numpy())
                        print(f'Epoch {epoch + 1} Batch {num_batches} Loss {loss_value.numpy()}')
                        
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', self.train_loss.result(), step=num_batches)
                        tf.summary.scalar('accuracy', self.train_accuracy.result(), step=num_batches)
                        tf.summary.scalar('auc', self.train_auc.result(), step=num_batches)

            self.evaluate(model, test_dataset,test_summary_writer)
            
            # save model            
            if not os.path.exists(self.config_xl.model_save_dir):
                os.makedirs(self.config_xl.model_save_dir)
            model_saved_dir =self.config_xl.model_save_dir+'/'+current_time+'_{}ep_{}mem_{}.ckpt/my_checkpoint'.format(self.config_xl.epoch, self.config_xl.mem_len, self.config_xl.mode)       
            model.save_weights(model_saved_dir)
            # self.config_xl.save_pretrained(self.config_xl.model_save_dir+'/'+current_time+'_{}ep_{}mem_{}.ckpt'.format(self.config_xl.epoch, self.config_xl.mem_len, self.config_xl.mode))

            logging.info('model_save_dir : %s',model_saved_dir)
            logging.info('model.summary: %s',model.summary()) 

        except Exception as e:
            logging.error(f"Error: {e}")

        return model
    
    def run(self):
        train_dataset,test_dataset,dkeyid2idx=self.load_TFdataset()
        train_summary_writer, test_summary_writer = self.make_tensorboard_summary_writer()
        model = self.train_test(train_dataset,test_dataset,train_summary_writer,test_summary_writer)
        self.Make_embedding_projector(model,dkeyid2idx)


def main(args) -> None :
    trainer = TransformerXLTrainer(args)
    trainer.run()



if __name__ == "__main__":

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
    parser.add_argument('--epoch', type=int, required=False, default=3)
    parser.add_argument('--mode', type=str, required=False, default='concepts',help='concepts or questions')
    parser.add_argument('--tf_data_dir', type=str, required=False, default='/home/jun/workspace/KT/data/ednet/TF_DATA1')
    parser.add_argument('--devices', type=str, required=False, default='gpu')
    parser.add_argument('--tensorboard_log_dir', type=str, required=False, default='/home/jun/workspace/KT/logs/gradient_tape')
    parser.add_argument('--tensorboard_emb_log_dir', type=str, required=False, default='/home/jun/workspace/KT/logs/embedding',help='tensorboard embedding projection dictionary')
    parser.add_argument('--model_save_dir', type=str, required=False, default='/home/jun/workspace/KT/save_model')
    args = parser.parse_args()

    if args.devices == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'


    main(args)