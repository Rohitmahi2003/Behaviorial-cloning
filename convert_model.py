import h5py
import json
import shutil
import os

def fix_model(input_path, output_path):
    # Make a temporary copy
    temp_path = "temp_fixed_model.h5"
    if os.path.exists(temp_path):
        os.remove(temp_path)
    shutil.copyfile(input_path, temp_path)
    
    # Open the temporary model file for editing
    with h5py.File(temp_path, 'r+') as f:
        # Get and fix the model_config
        model_config = f.attrs['model_config']
        config_json = json.loads(model_config.decode('utf-8'))
        
        modified = False
        
        # Process all layers
        for layer in config_json['config']['layers']:
            # Remove batch_shape from InputLayer
            if layer['class_name'] == 'InputLayer' and 'batch_shape' in layer['config']:
                print("Removing batch_shape parameter from InputLayer")
                del layer['config']['batch_shape']
                modified = True
                
            # Remove groups from Conv2D layers
            if layer['class_name'] == 'Conv2D' and 'groups' in layer['config']:
                print(f"Removing groups parameter from {layer['name']}")
                del layer['config']['groups']
                modified = True
                
        # Save fixed config back to file
        if modified:
            f.attrs['model_config'] = json.dumps(config_json).encode('utf-8')
    
    # Rename to final output
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(temp_path, output_path)
    print(f"Successfully created compatible model: {output_path}")

if __name__ == '__main__':
    print("Fixing model compatibility...")
    try:
        fix_model('model.h5', 'fully_compatible_model.h5')
    except Exception as e:
        print(f"Error during conversion: {str(e)}")