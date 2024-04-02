from torchvision import models, transforms
import torch
from PIL import Image
from captum.attr import DeepLift, Saliency
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading the model
model = models.googlenet(pretrained=True).to(device)
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Image paths and labels
image_info = {
    'black_swan': {'path': 'C:/Users/sontu/Desktop/NEU/Masters IoT/Sem 2/ML/HW4/Images/black_swan.jpg', 'label': 100},
    'daisy': {'path': 'C:/Users/sontu/Desktop/NEU/Masters IoT/Sem 2/ML/HW4/Images/daisy.jpg', 'label': 985},
    'golden_retriever': {'path': 'C:/Users/sontu/Desktop/NEU/Masters IoT/Sem 2/ML/HW4/Images/golden_retriever.jpg',
                         'label': 207},
    'goldfish': {'path': 'C:/Users/sontu/Desktop/NEU/Masters IoT/Sem 2/ML/HW4/Images/goldfish.jpg', 'label': 1},
    'hummingbird': {'path': 'C:/Users/sontu/Desktop/NEU/Masters IoT/Sem 2/ML/HW4/Images/hummingbird.jpg', 'label': 94}
}

# Initializing attribution methods
saliency = Saliency(model)
deeplift = DeepLift(model)


# Function to apply attributions and visualize
def apply_attributions(method, input_image, label, class_name):
    # Calculate attribution and convert to numpy array
    attribution = method.attribute(input_image, target=label).cpu().data.numpy()
    attribution = attribution.sum(axis=1)[0]
    attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min())

    plt.imshow(attribution, cmap='hot')
    plt.colorbar()
    plt.title(f'{class_name} - {method.__class__.__name__}')
    plt.axis('off')


# Processing and visualizing each image
for class_name, info in image_info.items():
    input_image = transform(Image.open(info['path']).convert('RGB')).unsqueeze(0).to(device)
    label = torch.tensor([info['label']], device=device)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(Image.open(info['path']))
    plt.title(f'Original Image - {class_name}')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    apply_attributions(saliency, input_image, label, class_name)

    plt.subplot(1, 3, 3)
    apply_attributions(deeplift, input_image, label, class_name)

    plt.show()