import torch
import torch.nn as nn

class FaultDiagnosis(nn.Module):

  def __init__(self):
    super().__init__()

    self.network = nn.Sequential(
        nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding='same'),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.3),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.3),

        nn.Conv2d(64, 128, kernel_size=3, padding='same'),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2)
    )

    self.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),

        nn.Flatten(),

        nn.Linear(in_features=128, out_features=64),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(in_features=64, out_features=32),
        nn.ReLU(),
        nn.Dropout(0.3),

        nn.Linear(in_features=32, out_features=4)
    )

  def forward(self, x):
    x = self.network(x)
    x = self.classifier(x)
    return x

CLASS_NAMES = [
    "Normal","Ball", "Inner" , "Outer"
]

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FaultDiagnosis().to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only = True))

    model.eval()

    return model, device

def predict(model, device, image_tensor):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        predicted_class = torch.argmax(probabilities, dim=1).item()

    return (
        CLASS_NAMES[predicted_class], 
        probabilities[0].cpu().numpy()
    )
