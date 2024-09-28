import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Step 1: Resize Images
def resize_images(input_dir, output_dir, size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to load image at path: {img_path}")
            continue
        img_resized = cv2.resize(img, size)
        cv2.imwrite(os.path.join(output_dir, img_name), img_resized)

# Example usage for resizing
resize_images(r'C:\Users\Prashanth\Desktop\Gender detection\dataset_directoryset_directory\Male', r'C:\Users\Prashanth\Desktop\Gender detection\Resized_Male_Images')
resize_images(r'C:\Users\Prashanth\Desktop\Gender detection\dataset_directory\Female', r'C:\Users\Prashanth\Desktop\Gender detection\Resized_Female_Images')

# Step 2: Data Augmentation
def augment_images(input_image_path, output_dir):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Unable to load image at path: {input_image_path}")
        return  # Exit the function if the image could not be loaded
    
    img = img.reshape((1,) + img.shape)  # Reshape image to match input shape
    i = 0
    for batch in datagen.flow(img, batch_size=1, save_to_dir=output_dir, save_prefix='aug', save_format='jpeg'):
        i += 1
        if i >= 20:  # Generate 20 augmented images and then stop
            break

# Example usage for augmentation
augment_images(r'C:\Users\Prashanth\Desktop\Gender detection\Resized_Male_Images\example_image.jpg', r'C:\Users\Prashanth\Desktop\Gender detection\Augmented_Male_Images')

# Step 3: Set Up the Model
def create_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)  # Sigmoid for binary classification (male/female)

    model = Model(inputs=base_model.input, outputs=x)
    return model

model = create_model()

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Load Data and Train the Model
def train_model(model, dataset_dir):
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=10,
        steps_per_epoch=train_generator.samples // 32,
        validation_steps=validation_generator.samples // 32
    )

    return history

# Example usage for training
history = train_model(model, r'C:\Users\Prashanth\Desktop\Gender detection\dataset_directory')

# Step 5: Evaluate the Model
def evaluate_model(model, dataset_dir):
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // 32)
    print(f'Test accuracy: {test_acc * 100:.2f}%')

# Example usage for evaluation
evaluate_model(model, r'C:\Users\Prashanth\Desktop\Gender detection\dataset_directory')

# Save the model
model.save('path_to_your_saved_model.h5')
print("Model saved successfully!")
