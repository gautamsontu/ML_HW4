from PIL import Image
import torchvision.transforms as transforms
import torch
from torchvision import models
from captum.attr import Saliency

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading the model
model = models.googlenet(pretrained=True)
model = model.to(device)
model.eval()

# Defining the image transformation: resize to 224x224, convert to tensor, and normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Image paths
image_paths = {
    'black_swan': r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW4\Images\black_swan.jpg',
    'daisy': r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW4\Images\daisy.jpg',
    'golden_retriever': r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW4\Images\golden_retriever.jpg',
    'goldfish': r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW4\Images\goldfish.jpg',
    'hummingbird': r'C:\Users\sontu\Desktop\NEU\Masters IoT\Sem 2\ML\HW4\Images\hummingbird.jpg'
}

# Labels for the classes according to ImageNet labels
imagenet_labels = {
    'black_swan': 100,
    'daisy': 985,
    'golden_retriever': 207,
    'goldfish': 1,
    'hummingbird': 94
}


# Function to load an image, transform it, and create predictions and saliency attributions
def predict_and_visualize(image_path, label_index):
    # Loading and transform the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    # Predicting the class
    output = model(image)
    prediction_score, pred_label_idx = torch.max(output, 1)
    predicted_label = pred_label_idx.item()

    # Initializing the Saliency object
    saliency = Saliency(model)

    # Getting saliency maps for the predicted label
    saliency_map_pred = saliency.attribute(image, target=predicted_label)
    saliency_map_pred = saliency_map_pred.squeeze().cpu().detach().numpy()

    # Getting saliency maps for the ground truth label
    saliency_map_true = saliency.attribute(image, target=label_index)
    saliency_map_true = saliency_map_true.squeeze().cpu().detach().numpy()

    return predicted_label, saliency_map_pred, saliency_map_true


# Dictionary to store results
results = {}

# Processing each image
for class_name, label_index in imagenet_labels.items():
    image_path = f'C:/Users/sontu/Desktop/NEU/Masters IoT/Sem 2/ML/HW4/Images/{class_name}.jpg'
    predicted_label, saliency_map_pred, saliency_map_true = predict_and_visualize(image_path, label_index)
    results[class_name] = {
        'predicted_label': predicted_label,
        'saliency_map_pred': saliency_map_pred,
        'saliency_map_true': saliency_map_true
    }

print(results)

for class_name, result in results.items():
    print(f"Class: {class_name}")
    print(f"Predicted Label Index: {result['predicted_label']}")
    print("Saliency Map for Predicted Label (summarized):")
    # Summarizing the saliency map by printing the max, min and mean values for each color channel
    print("  R Channel - Max: {:.6f}, Min: {:.6f}, Mean: {:.6f}".format(result['saliency_map_pred'][0].max(), result['saliency_map_pred'][0].min(), result['saliency_map_pred'][0].mean()))
    print("  G Channel - Max: {:.6f}, Min: {:.6f}, Mean: {:.6f}".format(result['saliency_map_pred'][1].max(), result['saliency_map_pred'][1].min(), result['saliency_map_pred'][1].mean()))
    print("  B Channel - Max: {:.6f}, Min: {:.6f}, Mean: {:.6f}".format(result['saliency_map_pred'][2].max(), result['saliency_map_pred'][2].min(), result['saliency_map_pred'][2].mean()))
    print("Saliency Map for Ground Truth Label (summarized):")
    # Summarizing the saliency map by printing the max, min and mean values for each color channel
    print("  R Channel - Max: {:.6f}, Min: {:.6f}, Mean: {:.6f}".format(result['saliency_map_true'][0].max(), result['saliency_map_true'][0].min(), result['saliency_map_true'][0].mean()))
    print("  G Channel - Max: {:.6f}, Min: {:.6f}, Mean: {:.6f}".format(result['saliency_map_true'][1].max(), result['saliency_map_true'][1].min(), result['saliency_map_true'][1].mean()))
    print("  B Channel - Max: {:.6f}, Min: {:.6f}, Mean: {:.6f}".format(result['saliency_map_true'][2].max(), result['saliency_map_true'][2].min(), result['saliency_map_true'][2].mean()))
    print("-" * 80)
