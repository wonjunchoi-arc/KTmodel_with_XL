import os
import json
import tensorflow as tf
import time
from models.model_for_kt import TFTransfoXLModel,TFTransfoXLLMHeadModel,TFTransfoXLMLMHeadModel
from transformers import TransfoXLConfig
from tensorflow.keras.utils import register_keras_serializable
from tqdm import tqdm
import datetime
import argparse


os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 실행코드에서 정하자



# parser = argparse.ArgumentParser(description='예시')
# parser.add_argument('--model_path', type=str, required=True)
# parser.add_argument('--audio_path', type=str, required=True)
# parser.add_argument('--transcript_path', type=str, required=True)
# parser.add_argument('--dst_path', type=str, required=True)
# parser.add_argument('--device', type=str, required=False, default='cpu')
# parser.add_argument('--device', type=str, required=False, default='cpu')
# parser.add_argument('--device', type=str, required=False, default='cpu')
# parser.add_argument('--device', type=str, required=False, default='cpu')
# parser.add_argument('--device', type=str, required=False, default='cpu')

# # opt는 parser로 나눈 모든 argument들을 dict 형식으로 가진다.
# opt = parser.parse_args()






config_xl = TransfoXLConfig(
    d_embed=128,
    d_head = 32,
    d_model=128,
    mem_len=600,
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
    Q_vocab_size = 12277,
    R_vocab_size = 2,
    epoch = 3,
    mode = 'questions', # concepts or questions 
    tf_dir = '/home/jun/workspace/KT/data/ednet/TF_DATA'
)

tf_train_dir = config_xl.tf_dir +'/train'
tf_test_dir = config_xl.tf_dir +'/test'
#dataset
if os.path.exists(tf_train_dir):
    train_dataset = tf.data.experimental.load(tf_train_dir)
    test_dataset = tf.data.experimental.load(tf_test_dir)
    with open(config_xl.tf_dir+"/keyid2idx.json", "r") as file:
        dkeyid2idx = json.load(file) 
else:
    train_dataset, test_dataset ,dkeyid2idx = make_dataset(config_xl)


#tensorboard logdir
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = '/home/jun/workspace/KT/logs/gradient_tape/' + current_time +'{}ep_{}mem_{}/train'.format(config_xl.epoch, config_xl.mem_len, config_xl.mode)
test_log_dir = '/home/jun/workspace/KT/logs/gradient_tape/' + current_time +'{}ep_{}mem_{}/test'.format(config_xl.epoch, config_xl.mem_len,config_xl.mode)
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


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
learning_rate = CustomSchedule(config_xl.d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model = TFTransfoXLMLMHeadModel(config=config_xl)

#Metric
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


def evaluate(model, mems, test_dataset):
    total_loss = 0.0
    num_batches = 0
    evaluation_metrics = []
    

    for test_question,test_ceq, test_mask, test_labels in test_dataset:
        input_test_data = test_ceq if config_xl.mode == 'concepts' else test_question
        outputs = model(concepts=input_test_data, responses=test_mask, labels=test_labels, mems=mems, training=False)
        loss = outputs.loss
        mems = outputs.mems

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


loss_values = []
num_batches = 0

for epoch in range(config_xl.epoch):
    start = time.time()
    total_loss = 0.0
    mems = None              # 첫 번째 모델의 메모리 상태           
    
    for question,ceq, mask, labels in tqdm(train_dataset):
        input_data = ceq if config_xl.mode == 'concepts' else question
        mems, loss_value = train_step(model, input_data,mask, labels, mems, optimizer)
        num_batches += 1
        total_loss += loss_value.numpy()
        if num_batches % 100 == 0:
            loss_values.append(loss_value.numpy())
            print(f'Epoch {epoch + 1} Batch {num_batches} Loss {loss_value.numpy()}')
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=num_batches)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=num_batches)
            tf.summary.scalar('auc', train_auc.result(), step=num_batches)

#save model weights and test model       

test_mems = None
test_loss0,test_acc0,average_precision, average_recall, average_f1_score = evaluate(model,test_mems, test_dataset)
print(f'Test Loss on First Half Dataset after Epoch {epoch + 1}: {test_loss0}')


#save model weights
model.save_weights('/home/jun/workspace/KT/save_model/{}ep_{}mem_{}.ckpt/my_checkpoint'.format(config_xl.epoch, config_xl.mem_len, config_xl.mode)) 
config_xl.save_pretrained('/home/jun/workspace/KT/save_model/{}ep_{}mem_{}.ckpt'.format(config_xl.epoch, config_xl.mem_len, config_xl.mode))



# make embedding projector 
from tensorboard.plugins import projector

log_dir='/home/jun/workspace/KT/logs/embedding/'+current_time+'{}ep_{}mem_{}/'.format(config_xl.epoch, config_xl.mem_len, config_xl.mode)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Save Labels separately on a line-by-line manner.
with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
  for concepts in dkeyid2idx['questions']:
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
 
# nohup python /home/jun/workspace/KT/train.py 1 > 1.out 2 > 2.out &