import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def create_compatible_model():
    """Create a TF 1.15 compatible NVIDIA model architecture"""
    model = Sequential()
    model.add(Conv2D(24, (5,5), strides=(2,2), input_shape=(66,200,3), activation='elu'))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation='elu'))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation='elu'))
    model.add(Conv2D(64, (3,3), activation='elu'))
    model.add(Conv2D(64, (3,3), activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    
    # Compile with dummy parameters (won't affect inference)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def copy_weights(original_model_path, new_model):
    """Copy weights from original model to new model"""
    try:
        # Load original model
        orig_model = load_model(original_model_path)
        
        # Copy weights layer by layer
        for new_layer, orig_layer in zip(new_model.layers, orig_model.layers):
            if new_layer.weights:
                new_layer.set_weights(orig_layer.get_weights())
                print(f"Copied weights for {new_layer.name}")
        return True
    except Exception as e:
        print(f"Failed to copy weights: {str(e)}")
        return False

if __name__ == '__main__':
    # Create new model
    model = create_compatible_model()
    
    # Copy weights from original model
    if copy_weights('model1.h5', model):
        print("Successfully copied weights from original model")
    else:
        print("Initializing with random weights")
    
    # Save the new compatible model
    model.save('compatible_model.h5', save_format='h5')
    print("Saved compatible model: compatible_model.h5")
    
    # Test loading
    try:
        test_model = load_model('compatible_model.h5')
        print("Test load successful!")
        print(test_model.summary())
    except Exception as e:
        print(f"Test load failed: {str(e)}")