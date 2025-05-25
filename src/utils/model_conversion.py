import numpy as np
import tensorflow as tf



# Function: Convert some hex value into an array for C programming
def hex_to_c_array(hex_data, var_name):
    c_str = ""

    # Create header guard
    c_str += "#ifndef " + var_name.upper() + "_H\n"
    c_str += "#define " + var_name.upper() + "_H\n\n"

    # Add array length at top of file
    c_str += "\nunsigned int " + var_name + "_len = " + str(len(hex_data)) + ";\n"

    # Declare C variable
    c_str += "unsigned char " + var_name + "[] = {"
    hex_array = []
    for i, val in enumerate(hex_data):
        # Construct string from hex
        hex_str = format(val, "#04x")

        # Add formatting so each line stays within 80 characters
        if (i + 1) < len(hex_data):
            hex_str += ","
        if (i + 1) % 12 == 0:
            hex_str += "\n "
        hex_array.append(hex_str)

    # Add closing brace
    c_str += "\n " + format(" ".join(hex_array)) + "\n};\n\n"

    # Close out header guard
    c_str += "#endif //" + var_name.upper() + "_H"

    return c_str


def generate_features_header(
    features: np.ndarray,
    vector_label: str,
    model_name: str,
    output_path: str,
    filename: str = "audio_features.h") -> None:
    """Generate a C++ header file containing audio features and quantization parameters.
    
    Args:
        features (np.ndarray): The extracted audio features (16 features)
        output_path (str): Path to save the header file
        filename (str): Name of the header file
    """
    
    print(">>> debug:", output_path)
    
    # Ensure features are in the correct shape (1, 16, 1)
    if features.shape != (1, 16, 1):
        features = features.reshape(1, 16, 1)
    
    # Get quantization parameters from the TFLite model
    interpreter = tf.lite.Interpreter(model_path=f"{output_path}/{model_name}.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    
    # Get quantization parameters
    scale = input_details['quantization_parameters']['scales'][0]
    zero_point = input_details['quantization_parameters']['zero_points'][0]
    
    # Quantize features
    quantized_features = np.round(features / scale + zero_point).astype(np.int8)
    
    # Generate C++ header content
    header_content = f"""#ifndef AUDIO_FEATURES_H
#define AUDIO_FEATURES_H

// Quantization parameters
const float input_scale = {scale}f;
const int input_zero_point = {zero_point};

// Quantized audio features
const int8_t {vector_label}[16] = {{
    {', '.join(str(x) for x in quantized_features.flatten())}
}};

#endif // AUDIO_FEATURES_H
"""
    
    # Write to file
    with open(f"{output_path}/{filename}", "w") as f:
        f.write(header_content)
    
    print(f"Audio features header generated at {output_path}/{vector_label}-{filename}")
