# train_facing_classifier.py
import os
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import mobilenet_v3
from tensorflow.keras.preprocessing import image_dataset_from_directory

def build_model(input_shape=(96,96,3), dropout=0.2):
    base = mobilenet_v3.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights=None  # train from scratch or set 'imagenet' if you want pretraining
    )
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=base.input, outputs=out)
    return model

def get_datasets(data_dir, img_size=(96,96), batch_size=64):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    train_ds = image_dataset_from_directory(train_dir,
                                            labels='inferred',
                                            label_mode='binary',
                                            image_size=img_size,
                                            batch_size=batch_size,
                                            shuffle=True)
    val_ds = image_dataset_from_directory(val_dir,
                                          labels='inferred',
                                          label_mode='binary',
                                          image_size=img_size,
                                          batch_size=batch_size,
                                          shuffle=False)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE).map(lambda x,y: (x/255.0, y))
    val_ds = val_ds.prefetch(AUTOTUNE).map(lambda x,y: (x/255.0, y))
    return train_ds, val_ds

def main(args):
    tf.random.set_seed(42)
    train_ds, val_ds = get_datasets(args.data_dir, img_size=(args.img_size, args.img_size), batch_size=args.batch_size)

    model = build_model(input_shape=(args.img_size, args.img_size, 3), dropout=args.dropout)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.save_model_path, save_best_only=True, monitor='val_accuracy', mode='max'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    ]

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=args.epochs,
                        callbacks=callbacks)

    print("Training finished. Best model saved to:", args.save_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset", help="dataset root containing train/val subfolders")
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--save_model_path", type=str, default="facing_model.h5")
    args = parser.parse_args()
    main(args)
