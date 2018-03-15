import tensorflow as tf

# Default hyperparameters:
hparams = tf.contrib.training.HParams(
    num_mels=128,
    num_freq=1013,
    sample_rate=16000,
    frame_length_ms=16.0,
    frame_shift_ms=8.0,
    preemphasis=0.97,
    min_level_db=-80,
    ref_level_db=20,

    griffin_lim_iters=60
)

def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
