import base64
import os
from io import BytesIO

import torch
from esun.domain.abstract_entity import AbstractEntity
from esun.domain.abstract_id import AbstractId
from esun.domain.image_class_index import idx_to_class
from PIL import Image
from torchvision import transforms

DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {DEVICE}')


class Id(AbstractId):

    def __init__(self, value: str) -> None:
        super().__init__(value=value)


class ImageClassifier(AbstractEntity):
    MODEL = torch.load(
        os.environ['MODEL_PATH'], map_location=DEVICE)
    MODEL.eval()
    TRANSFORMS = transforms.Compose([
        transforms.Resize(140),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=3),
    ])
    LABEL_NAMES = idx_to_class

    def __init__(self, id: str, image_64_encoded: str) -> None:
        super().__init__(id=Id(value=id))
        self._size = (224, 224)
        self._image_64_encoded = image_64_encoded

    def get_answer(self) -> str:
        image = self._get_image()
        with torch.no_grad():
            image_tensor = ImageClassifier.TRANSFORMS(image)
            image_tensor = torch.unsqueeze(image_tensor, 0)
            outputs = ImageClassifier.MODEL(image_tensor)
            _, preds = torch.max(outputs, 1)
            class_idx = preds[0].item()
            class_name = ImageClassifier.LABEL_NAMES[class_idx]
            return class_name

    def _get_image(self):
        img_base64_binary = self._image_64_encoded.encode("utf-8")
        img_binary = base64.b64decode(img_base64_binary)
        image = Image.open(BytesIO(img_binary))
        return image
