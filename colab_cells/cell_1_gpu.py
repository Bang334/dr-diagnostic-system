# ════════════════════════════════════════════════════════════════
# Cell 1: Kiểm tra GPU
# ════════════════════════════════════════════════════════════════
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    raise RuntimeError('Hãy chọn Runtime > Change runtime type > T4 GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass
print('TensorFlow:', tf.__version__)
print('GPU:', gpus)
