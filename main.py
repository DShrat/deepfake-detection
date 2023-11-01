# tambahan
from collections import Counter

# webserver part
from fastapi import FastAPI, UploadFile, File

# cnn part
import tensorflow as tf
import numpy as np
import cv2
from imgaug import augmenters as iaa

# webserver code start
app = FastAPI()
# webserver code end

# AI code
resnet50 = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')
modelR = tf.keras.Model(inputs=resnet50.input, outputs=resnet50.get_layer('conv4_block5_out').output)

xception = tf.keras.applications.xception.Xception(include_top=False, weights='imagenet')
modelX = tf.keras.Model(inputs=xception.input, outputs=xception.get_layer('conv2d_2').output)

vgg19 = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
modelV = tf.keras.Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)

checkpoint_path_R = "./resnet/my_capsule_network"  
checkpoint_path_X = "./xception/my_capsule_network"  
checkpoint_path_V = "./vgg/my_capsule_network"  

def preprocess_image(img):
    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Define augmentation sequence
    augmenter = iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)), 
    ])

    # # Apply the augmentation to the image
    img_array = augmenter.augment_image(img_array)

    # Resize the augmented image to a specific size (e.g., 224x224 pixels)
    img_array = cv2.resize(img_array, (224, 224))

    # Perform any other preprocessing steps like normalization (if needed)

    return img_array

#RESNET50
def try_predict_resnet(filepath):
    image = cv2.imread(filepath)
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    features = modelR.predict(image)
    combined_feature = np.mean(features, axis=-1, keepdims=True) #1 layer
    
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph(checkpoint_path_R + '.meta')
        saver.restore(sess, checkpoint_path_R)
        graph = tf.compat.v1.get_default_graph()
        X = graph.get_tensor_by_name("X:0")  # Sesuaikan dengan nama placeholder

        # prediksi pada data X_test
        predictions = graph.get_tensor_by_name("y_pred:0")  # Ganti nama tensor output 
        predicted_labels = sess.run(predictions, feed_dict={X: combined_feature.reshape(1, 14, 14, 1)})

    # predicted_labels sekarang akan berisi hasil prediksi untuk data X_test
    # return {str(predicted_labels)}
    if predicted_labels == 0:
        return {"real"}
    elif predicted_labels == 1:
        return {"deepfake"}
    else:
        return {"gagal mendeteksi label, hasilnya ": predictions}
    
#VGG19
def try_predict_vgg(filepath):
    image = cv2.imread(filepath)
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.vgg19.preprocess_input(image)
    features = modelV.predict(image)
    combined_feature = np.mean(features, axis=-1, keepdims=True) #1 layer
    
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph(checkpoint_path_V + '.meta')
        saver.restore(sess, checkpoint_path_V)
        graph = tf.compat.v1.get_default_graph()
        X = graph.get_tensor_by_name("X:0")  # Sesuaikan dengan nama placeholder

        # prediksi pada data X_test
        predictions = graph.get_tensor_by_name("y_pred:0")  # Ganti nama tensor output 
        predicted_labels = sess.run(predictions, feed_dict={X: combined_feature.reshape(1, 14, 14, 1)})

    # predicted_labels sekarang akan berisi hasil prediksi untuk data X_test
    if predicted_labels == 0:
        return {"real"}
    elif predicted_labels == 1:
        return {"deepfake"}
    else:
        return {"gagal mendeteksi label, hasilnya ": predictions}
    
#XCEPTION
def try_predict_xception(filepath):
    image = cv2.imread(filepath)
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.xception.preprocess_input(image)
    features = modelX.predict(image)
    combined_feature = np.mean(features, axis=-1, keepdims=True) #1 layer
    
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph(checkpoint_path_X + '.meta')
        saver.restore(sess, checkpoint_path_X)
        graph = tf.compat.v1.get_default_graph()
        X = graph.get_tensor_by_name("X:0")  # Sesuaikan dengan nama placeholder

        # prediksi pada data X_test
        predictions = graph.get_tensor_by_name("y_pred:0")  # Ganti nama tensor output 
        predicted_labels = sess.run(predictions, feed_dict={X: combined_feature.reshape(1, 14, 14, 1)})

    # predicted_labels sekarang akan berisi hasil prediksi untuk data X_test
    if predicted_labels == 0:
        return {"real"}
    elif predicted_labels == 1:
        return {"deepfake"}
    else:
        return {"gagal mendeteksi label, hasilnya ": predictions}

#count
def find_majority_vote(items):
    # Count the occurrences of each item in the list
    item_counts = Counter(items)

    # Find the item with the maximum count
    majority_item, majority_count = item_counts.most_common(1)[0]

    # Check if the majority count is greater than half the length of the list
    if majority_count > len(items) / 2:
        return majority_item
    else:
        return "No majority vote"

#voting
def voting(filepath):
    hasilV = str(try_predict_vgg(filepath))
    hasilR = str(try_predict_resnet(filepath))
    hasilX = str(try_predict_xception(filepath))

    items = []
    items.append(hasilV)
    items.append(hasilR)
    items.append(hasilX)

    majority_vote = find_majority_vote(items)

    if majority_vote == "{'real'}":
        return {"result":"real"}
    elif majority_vote == "{'deepfake'}":
        return {"result":"deepfake"}
    else:
        return {"tidak dapat mendeteksi"}

# AI code done

@app.get("/hello-world")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(file: UploadFile):
    if file.filename.endswith(".jpg") or file.filename.endswith(".jpeg"):
        file_location = f"public/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

        return voting(file_location)
    else:
        {"error": "invalid image type, must be jpg or jpeg"} 
    return {"filename": file.filename}

    # return try_predict('public/test-1.jpg')
