import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image

# Load the pre-trained ResNet50 model for feature extraction
def extract_features(image_path):
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array, verbose=0)
    return features.flatten()

# Load the tokenizer (either create or load a pre-trained tokenizer)
def load_tokenizer(captions_file='captions.txt'):
    # Load the captions and fit the tokenizer
    with open(captions_file, 'r') as file:
        captions = file.readlines()
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    return tokenizer

# Generate a caption for a given image using the model and tokenizer
def generate_caption(model, img_feature, tokenizer, max_sequence_length=34):
    caption = 'startseq'
    for i in range(max_sequence_length):
        sequence = tokenizer.texts_to_sequences([caption])
        sequence = pad_sequences(sequence, maxlen=max_sequence_length)
        
        # Predict the next word in the sequence
        predicted_probs = model.predict([img_feature, sequence], verbose=0)
        predicted_word_index = np.argmax(predicted_probs)
        
        # Get the word from the predicted index
        predicted_word = tokenizer.index_word.get(predicted_word_index, None)
        
        if predicted_word is None:
            break
        
        caption += ' ' + predicted_word
        
        if predicted_word == 'endseq':
            break
            
    return caption

# Load the model
def load_captioning_model():
    model = load_model('image_captioning_model.h5')  # Path to your saved model
    return model

# Main function to generate a caption for a given image
def main():
    image_directory = r"D:\SRM\CODE\images"  # Path to your image directory
    captions_file = 'D:/SRM/CODE/captions.txt'  # Path to your captions file

    # Load the model and tokenizer
    model = load_captioning_model()
    tokenizer = load_tokenizer(captions_file)

    # Example: Process a single image
    image_path = r'D:\SRM\CODE\images\test_image.jpg' # Path to the test image
    img_feature = extract_features(image_path)
    
    # Generate caption
    caption = generate_caption(model, img_feature, tokenizer)
    
    print(f'Generated Caption: {caption}')
    
    # Optionally, display the image with the caption
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(caption)
    plt.show()

if __name__ == "__main__":
    main()
