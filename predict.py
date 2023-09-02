import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import densenet121
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class cifar100:
    def __init__(self,filename):
        self.filename =filename
        self.filepath="cifar100_classes.txt"

    def load_cifar100_classes(self,file_path):
        with open(file_path, "r") as file:
            cifar100_classes = [line.strip() for line in file]
        return cifar100_classes

    def predictioncifar100(self):
        # load model
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])

        # Load the saved model
        model = densenet121(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Linear(1024, 500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(250, 100),
            nn.LogSoftmax(dim=1)
        )

        # Load the model on CPU regardless of CUDA availability
        model.load_state_dict(torch.load('dense_new.pth', map_location=torch.device('cpu')))
        model.to(device)
        model.eval()

        # summarize model
        #model.summary()
        imagename = self.filename
        image = Image.open(imagename)
        image = transform(image).unsqueeze(0).to(device)  # Add an extra dimension for batch size

        # Apply the model to the image
        with torch.no_grad():
            output = model(image)
            probabilities = torch.exp(output)
            _, predicted = torch.max(output.data, 1)
            class_index = predicted.item()
            prediction_score =  f"{(probabilities[0, class_index].item() * 100):.2f}"

        # file_path = "cifar100_classes.txt"
        cifar100_classes = self.load_cifar100_classes(self.filepath)
        if class_index >= 0 and class_index < len(cifar100_classes):
            class_name = cifar100_classes[class_index]
            print(f"Class score {prediction_score} corresponds to class name: {class_name}")
        else:
            print(f"Invalid class index: {class_index}")
        result={
            "prediction_score":str(prediction_score),
            "class_name":class_name

        }
        return result


