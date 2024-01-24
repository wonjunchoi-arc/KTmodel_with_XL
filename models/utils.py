import tensorflow as tf

def match_seq_len_tf(q_seqs, r_seqs, seq_len, pad_val=-1):
    '''
    TensorFlow 버전의 match_seq_len 함수.
    Args:
        q_seqs, r_seqs: 질문 시퀀스와 응답 시퀀스의 리스트
        seq_len: 맞추려는 시퀀스 길이
        pad_val: 패딩에 사용할 값
    Returns:
        proc_q_seqs, proc_r_seqs: 처리된 시퀀스
    '''
    proc_q_seqs = tf.keras.preprocessing.sequence.pad_sequences(
        q_seqs, maxlen=seq_len, padding='post', truncating='post', value=pad_val
    )
    proc_r_seqs = tf.keras.preprocessing.sequence.pad_sequences(
        r_seqs, maxlen=seq_len, padding='post', truncating='post', value=pad_val
    )

    return proc_q_seqs, proc_r_seqs

def collate_fn_tf(batch, pad_val=-1):
    '''
    TensorFlow 버전의 collate_fn 함수.
    Args:
        batch: 배치 데이터 (q_seqs, r_seqs 쌍의 리스트)
        pad_val: 패딩에 사용할 값
    Returns:
        처리된 텐서
    '''
    q_seqs, r_seqs = zip(*batch)

    q_seqs, r_seqs = match_seq_len_tf(q_seqs, r_seqs, max(map(len, q_seqs)), pad_val)

    qshft_seqs = tf.roll(q_seqs, shift=-1, axis=1)
    rshft_seqs = tf.roll(r_seqs, shift=-1, axis=1)

    mask_seqs = tf.cast(tf.not_equal(q_seqs, pad_val), tf.float32)

    q_seqs, r_seqs, qshft_seqs, rshft_seqs = (
        q_seqs * mask_seqs, r_seqs * mask_seqs, qshft_seqs * mask_seqs, rshft_seqs * mask_seqs
    )

    return q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs

# 예제 코드는 TensorFlow의 함수들을 사용하여 PyTorch 코드의 기능을 재현합니다.
# 실제 환경에서 TensorFlow 2.x가 설치된 상태에서 이 코드를 실행할 수 있습니다.

