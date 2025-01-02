import torch
import cv2
import numpy as np
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

# Initialize the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model weights
weights = torch.load('C:/Users/shapi/OneDrive/Desktop/Projects/005_fall_detection/yolov7-w6-pose.pt', map_location=device)
model = weights['model']
_ = model.float().eval()

if torch.cuda.is_available():
    model.half().to(device)

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Loop for real-time processing of frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the image
    image = letterbox(frame, 960, stride=64, auto=True)[0]  # Resize to model input size
    image_ = image.copy()
    image = transforms.ToTensor()(image)  # Convert to tensor
    image = torch.tensor(np.array([image.numpy()]))  # Add batch dimension

    if torch.cuda.is_available():
        image = image.half().to(device)  # Move image to GPU and use half precision
    
    # Run inference
    with torch.no_grad():
        output, _ = model(image)
    
    # Apply non-max suppression
    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)

    # Convert output to keypoints
    output = output_to_keypoint(output)

    # Convert tensor to image format for visualization
    nimg = image[0].permute(1, 2, 0) * 255  # Convert to HxWx3
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR for OpenCV

    # Plot the keypoints on the image
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

    # Convert back to RGB before showing
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    
    # Show the processed frame
    cv2.imshow('Pose Estimation', nimg)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
