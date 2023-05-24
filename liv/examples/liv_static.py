import clip
import torch
import torchvision.transforms as T
from PIL import Image 

from liv import load_liv

device = "cuda" if torch.cuda.is_available() else "cpu"

# loading LIV
liv = load_liv()
liv.eval()
transform = T.Compose([T.ToTensor()])

# pre-process image and text
image = transform(Image.open("sample_video/frame_0000033601.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["open microwave", "close microwave", "wipe floor"]).to(device)

# compute LIV image and text embedding
with torch.no_grad():
    img_embedding = liv(input=image, modality="vision")
    text_embedding = liv(input=text, modality="text")

# compute LIV value
img_text_value = liv.module.sim(img_embedding, text_embedding)
# Output: [ 0.1151, -0.0151, -0.0997]