import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Thiết lập các tham số
batch_size = 32
epochs = 10
image_size = (32, 32)
num_classes = 62  # Số lượng lớp ký tự trong tập dữ liệu Chars74K

# Đường dẫn tới thư mục chứa tập dữ liệu Chars74K
dataset_dir = 'char74ks'

# Tạo các đối tượng ImageDataGenerator để tạo dữ liệu huấn luyện và kiểm tra từ thư mục
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Chia tập dữ liệu thành tập huấn luyện và tập validation
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Sử dụng tập huấn luyện
)

validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Sử dụng tập validation
)

# Xây dựng mô hình CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Biên soạn và huấn luyện mô hình
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Đánh giá mô hình
test_loss, test_acc = model.evaluate(validation_generator)
print("Test accuracy:", test_acc)
model.save('char74k.h5')