
import os, random, math, json
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_curve, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")

DATASET_PATH = r"T:\DeepFake\dataset\DATASET"   
IMG_SIZE = 160      
BATCH_SIZE = 32
EPOCHS = 18
SEED = 42
TTA_ROUNDS = 2     
OUTPUT_JSON = "ythrinesh_prediction.json"


random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

REAL_DIR = os.path.join(DATASET_PATH, r"T:\DeepFake\dataset\DATASET\real_cifake_images")
FAKE_DIR = os.path.join(DATASET_PATH, r"T:\DeepFake\dataset\DATASET\fake_cifake_images")
TEST_DIR = os.path.join(DATASET_PATH, r"T:\DeepFake\dataset\DATASET\test")

for p in [REAL_DIR, FAKE_DIR, TEST_DIR]:
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Missing folder: {p}")


def compute_feats_np(img_rgb_uint8):
    gray = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    lap_var = np.var(lap)
    sobel_var = np.var(np.sqrt(sx*sx + sy*sy))
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    hist_norm = hist / (hist.sum() + 1e-9)
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-9))
    return np.array([lap_var, sobel_var, entropy], dtype=np.float32)

def load_and_resize(path):
    img = cv2.imread(path)
    if img is None:
        img = np.zeros((IMG_SIZE,IMG_SIZE,3), dtype=np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def list_images(folder):
    exts = (".png", ".jpg", ".jpeg")
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    try:
        files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
    except:
        files = sorted(files)
    return files

real_files = list_images(REAL_DIR)
fake_files = list_images(FAKE_DIR)
test_files = list_images(TEST_DIR)

print("Counts -> real:", len(real_files), "fake:", len(fake_files), "test:", len(test_files))


train_paths = [os.path.join(REAL_DIR,f) for f in real_files] + [os.path.join(FAKE_DIR,f) for f in fake_files]
train_labels = [1.0]*len(real_files) + [0.0]*len(fake_files)


paths, labels = train_paths, train_labels
paths, labels = np.array(paths), np.array(labels)
idxs = np.arange(len(paths))
np.random.shuffle(idxs)
paths, labels = paths[idxs].tolist(), labels[idxs].tolist()

train_paths_split, val_paths_split, y_train, y_val = train_test_split(paths, labels, test_size=0.18, random_state=SEED, stratify=labels)


all_paths_for_feats = train_paths_split + val_paths_split + [os.path.join(TEST_DIR, f) for f in test_files]
print("Computing handcrafted features for", len(all_paths_for_feats), "images (this may take a while)...")
all_feats = []
for p in tqdm(all_paths_for_feats):
    img = load_and_resize(p)
    feats = compute_feats_np(img)
    all_feats.append(feats)
all_feats = np.stack(all_feats, axis=0)


n_train = len(train_paths_split)
scaler = StandardScaler()
scaler.fit(all_feats[:n_train])
scaled_feats = scaler.transform(all_feats)


train_feats = scaled_feats[:len(train_paths_split)]
val_feats = scaled_feats[len(train_paths_split):len(train_paths_split)+len(val_paths_split)]
test_feats = scaled_feats[len(train_paths_split)+len(val_paths_split):]


def make_dataset_from_arrays(paths_list, feats_array, labels_list=None, training=True):
    paths_tensor = tf.constant(paths_list)
    feats_tensor = tf.constant(feats_array, dtype=tf.float32)
    if labels_list is None:
        ds = tf.data.Dataset.from_tensor_slices((paths_tensor, feats_tensor))
    else:
        labels_tensor = tf.constant(labels_list, dtype=tf.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths_tensor, feats_tensor, labels_tensor))
    if training and labels_list is not None:
        ds = ds.shuffle(len(paths_list), seed=SEED)
    
    def _map_with_label(path, feat, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        return ((img, feat), label)

    def _map_no_label(path, feat):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        return ((img, feat),)   

    if labels_list is not None:
        ds = ds.map(_map_with_label, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            def aug(data, label):
                (img, feat) = data
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_brightness(img, 0.10)
                img = tf.image.random_contrast(img, 0.9, 1.1)
                return ((img, feat), label)
            ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.repeat()
    else:
        ds = ds.map(_map_no_label, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset_from_arrays(train_paths_split, train_feats, y_train, training=True)
val_ds = make_dataset_from_arrays(val_paths_split, val_feats, y_val, training=False)
test_ds = make_dataset_from_arrays([os.path.join(TEST_DIR,f) for f in test_files], test_feats, labels_list=None, training=False)

def cnn_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(2)(x)
    return x

def build_model_variant_A():
    img_in = Input((IMG_SIZE, IMG_SIZE, 3))
    feat_in = Input((3,))
    x = cnn_block(img_in, 32)
    x = cnn_block(x, 64)
    x = cnn_block(x, 96)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    f = layers.Dense(32, activation='relu')(feat_in)
    f = layers.BatchNormalization()(f)
    merged = layers.concatenate([x, f])
    merged = layers.Dense(128, activation='relu')(merged)
    merged = layers.Dropout(0.4)(merged)
    out = layers.Dense(1, activation='sigmoid')(merged)
    model = Model([img_in, feat_in], out)
    return model

def build_model_variant_B():
    img_in = Input((IMG_SIZE, IMG_SIZE, 3))
    feat_in = Input((3,))
    x = cnn_block(img_in, 48)
    x = cnn_block(x, 96)
    x = cnn_block(x, 128)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(192, activation='relu')(x)
    x = layers.Dropout(0.45)(x)

    f = layers.Dense(48, activation='relu')(feat_in)
    f = layers.BatchNormalization()(f)
    merged = layers.concatenate([x, f])
    merged = layers.Dense(192, activation='relu')(merged)
    merged = layers.Dropout(0.45)(merged)
    out = layers.Dense(1, activation='sigmoid')(merged)
    model = Model([img_in, feat_in], out)
    return model

def build_model_variant_C():
    img_in = Input((IMG_SIZE, IMG_SIZE, 3))
    feat_in = Input((3,))
    x = cnn_block(img_in, 32)
    x = cnn_block(x, 64)
    x = cnn_block(x, 96)
    x = cnn_block(x, 128)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    f = layers.Dense(64, activation='relu')(feat_in)
    f = layers.BatchNormalization()(f)
    merged = layers.concatenate([x, f])
    merged = layers.Dense(256, activation='relu')(merged)
    merged = layers.Dropout(0.5)(merged)
    out = layers.Dense(1, activation='sigmoid')(merged)
    model = Model([img_in, feat_in], out)
    return model

models = []
val_preds = []   
test_preds = []  
builders = [build_model_variant_A, build_model_variant_B, build_model_variant_C]

for i, builder in enumerate(builders):
    print(f"\n=== Training model {i+1}/{len(builders)} ===")
    model = builder()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy', metrics=['accuracy'])
    steps_per_epoch = max(1, len(train_paths_split)//BATCH_SIZE)
    validation_steps = max(1, len(val_paths_split)//BATCH_SIZE)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]

    model.fit(train_ds,
              epochs=EPOCHS,
              steps_per_epoch=steps_per_epoch,
              validation_data=val_ds,
              validation_steps=validation_steps,
              callbacks=callbacks,
              verbose=2)


    val_pred = model.predict(val_ds, verbose=0).ravel()[:len(y_val)]
    val_preds.append(val_pred)


    test_pred = model.predict(test_ds, verbose=0).ravel()[:len(test_files)]
    test_preds.append(test_pred)

    models.append(model)


X_val_stack = np.vstack(val_preds).T 
y_val_arr = np.array(y_val, dtype=np.float32)
stacker = Ridge(alpha=1.0)
stacker.fit(X_val_stack, y_val_arr)
stack_val_pred = stacker.predict(X_val_stack)
stack_test_pred = stacker.predict(np.vstack(test_preds).T)


iso = IsotonicRegression(out_of_bounds='clip')
try:
    iso.fit(stack_val_pred, y_val_arr)
    calibrated_val = iso.transform(stack_val_pred)
    calibrated_test = iso.transform(stack_test_pred)
    use_iso = True
    print("Isotonic calibration applied.")
except Exception as e:
    print("Isotonic calibration failed — skipping. Error:", e)
    calibrated_val = stack_val_pred
    calibrated_test = stack_test_pred
    use_iso = False


fpr, tpr, thr = roc_curve(y_val_arr, calibrated_val)
youden = tpr - fpr
best_idx = youden.argmax()
best_threshold = thr[best_idx]
val_acc = ((calibrated_val >= best_threshold).astype(int) == y_val_arr.astype(int)).mean()
print(f"Adaptive threshold (Youden J): {best_threshold:.4f}, Val accuracy (after calibration): {val_acc*100:.2f}%")


def simple_tta_predict_all(models, test_file_list):
    
    per_model_preds = []
    for m in models:
        preds = []
       
        preds_orig = m.predict(test_ds, verbose=0).ravel()[:len(test_file_list)]
        
        imgs = []
        feats = []
        for p in test_file_list:
            img = load_and_resize(os.path.join(TEST_DIR, p))
            img_f = img[:, ::-1, :]
            imgs.append(img_f.astype(np.float32)/255.0)
            feats.append(compute_feats_np(img_f.astype(np.uint8)))
        imgs = np.array(imgs, dtype=np.float32)
        feats = np.array(scaler.transform(feats), dtype=np.float32)  
        preds_flip = m.predict([imgs, feats], verbose=0).ravel()[:len(test_file_list)]
      
        preds_avg = 0.5 * (preds_orig + preds_flip)
        per_model_preds.append(preds_avg)
    return np.vstack(per_model_preds)  
try:
    tta_stack = simple_tta_predict_all(models, test_files)
    tta_stack_stacked = stacker.predict(tta_stack.T)
    if use_iso:
        tta_stack_stacked = iso.transform(tta_stack_stacked)
   
    final_test_scores = 0.6 * calibrated_test + 0.4 * tta_stack_stacked
    final_test_scores = np.clip(final_test_scores, 0.0, 1.0)
    print("TTA applied and blended into final test scores.")
except Exception as e:
    print("TTA failed or skipped due to error:", e)
    final_test_scores = np.clip(calibrated_test, 0.0, 1.0)

results = []
for fname, score in zip(test_files, final_test_scores):
    try:
        idx = int(os.path.splitext(fname)[0])
    except:
        idx = fname
    results.append({"index": idx, "prediction": float(score)})

def idx_key(x):
    k = x["index"]
    return int(k) if isinstance(k, (int, np.integer)) or (isinstance(k, str) and k.isdigit()) else str(k)

results = sorted(results, key=idx_key)

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=4)
print("Saved final predictions to", OUTPUT_JSON)
print("Sample outputs:", results[:5])



