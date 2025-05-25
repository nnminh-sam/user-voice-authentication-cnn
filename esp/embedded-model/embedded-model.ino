#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Best model: model1.h - description: model1.py
#include "model.h"


// Tensor arena size (50 KB) - may need adjustment based on model requirements
constexpr int kTensorArenaSize = 65 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// TensorFlow Lite objects
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// Feature arrays corrected to float type
const float anh_ban_than_features[] = {
    -419.49951171875, 93.13069152832031, 16.134830474853516, -1.3128151893615723,
    11.915136337280273, -13.64680290222168, -14.209481239318848, -16.481754302978516,
    -4.970718860626221, -14.792462348937988, 2.2262680530548096, -0.12527583539485931,
    4000.0, 0.00898216012865305, 760.320556640625, 0.07653223016605167
};
const int anh_ban_than_features_size = 16;

const float giang_oi_features[] = {
    -329.1415100097656, 69.21592712402344, 7.7742509841918945, 7.717978477478027,
    -2.0862982273101807, 0.925134539604187, -15.983000755310059, -21.052215576171875,
    -8.66250228881836, -22.550331115722656, -13.991190910339355, -6.666601657867432,
    4000.0, 0.011978049762547016, 1366.4332275390625, 0.09616497064092357
};
const int giang_oi_features_size = 16;

void setup() {
    Serial.begin(115200);

    // Load the model from flash memory
    model = tflite::GetModel(voice_identification);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema mismatch!");
        return;
    }

    // Initialize interpreter with corrected error reporter
    interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);

    // Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("AllocateTensors() failed - increase kTensorArenaSize if this persists");
        return;
    }

    // Optional: Check how much of the tensor arena is used
    Serial.print("Tensor arena used bytes: ");
    Serial.println(interpreter->arena_used_bytes());

    // Get pointers to input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
}

void predict(const float* features, size_t feature_size, const char* message) {
    Serial.println(message);

    // Copy features into the input tensor (assuming float32 input)
    for (size_t i = 0; i < feature_size; ++i) {
        input->data.f[i] = features[i];
    }

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke failed");
        return;
    }

    // Interpret output (assuming two classes: "Anh ban than" and "Giang oi")
    float prob_anh_ban_than = output->data.f[0];
    float prob_giang_oi = output->data.f[1];
    if (prob_anh_ban_than > prob_giang_oi) {
        Serial.println("Predicted: Anh ban than");
    } else {
        Serial.println("Predicted: Giang oi");
    }
    Serial.print("Probabilities - Anh ban than: ");
    Serial.print(prob_anh_ban_than);
    Serial.print(", Giang oi: ");
    Serial.println(prob_giang_oi);
    Serial.println();
}

void loop() {
    Serial.println("Predicting...");
    delay(500);

    predict(anh_ban_than_features, anh_ban_than_features_size, "Predicting using [Anh ban than] features");
    predict(giang_oi_features, giang_oi_features_size, "Predicting using [Giang oi] features");
}