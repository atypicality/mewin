import cv2
import numpy as np
import tensorflow as tf

# 1. Load the model and labels
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Get model requirements
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
is_quantized = input_details[0]['dtype'] == np.uint8

# 2. Open the Mac webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Process the frame to match what the model expects
    img = cv2.resize(frame, (width, height))
    img = np.expand_dims(img, axis=0)
    
    if not is_quantized:
        img = (np.float32(img) / 127.5) - 1.0

    # 4. Predict
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # 5. Display the highest confidence result
    top_index = np.argmax(predictions)
    confidence = predictions[top_index] / 255.0 if is_quantized else predictions[top_index]
    label_text = f"{labels[top_index]}: {confidence:.0%}"

    cv2.putText(frame, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Mac Trash Classifier", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()