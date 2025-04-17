import os
import numpy as np
import joblib
from sklearn.svm import SVC
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

# Step 1: Load dataset
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = datagen.flow_from_directory(
    'brain_tumor_dataset',
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse',
    shuffle=False
)

# Step 2: Extract features
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(150, 150, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

features = feature_extractor.predict(train_generator)
labels = train_generator.classes

# Step 3: Train SVM
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(features, labels)

# Step 4: Create label map
label_map = train_generator.class_indices
reverse_label_map = {v: k.replace("_", " ").title() + " Tumor" if k != 'no' else "No Tumor"
                     for k, v in label_map.items()}

# Step 5: Save model and label map together
save_dict = {
    'model': svm_model,
    'label_map': reverse_label_map
}
joblib.dump(save_dict, 'model.pkl')  # 👈 Overwrite model.pkl
print("✅ Model and label map saved in 'model.pkl'")
