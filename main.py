import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as transforms
from model import SSD300, ResNet, Loss
from train import train_loop, tencent_trick
from PIL import ImageMath
import cv2

#Sources : https://github.com/julimueller/dtld_parsing
# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD

preprocessFn = transforms.Compose(
    [transforms.Resize((300,300)),  
     transforms.ToTensor(), 
     transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                          std = [0.229, 0.224, 0.225])])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
"""
precision = 'fp32'
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')


ssd_model.to(device)
ssd_model.eval()

uris = [
    'http://images.cocodataset.org/val2017/000000397133.jpg',
    'http://images.cocodataset.org/val2017/000000037777.jpg',
    'http://images.cocodataset.org/val2017/000000252219.jpg'
]

inputs = [utils.prepare_input(uri) for uri in uris]
"""
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
img = cv2.imread("data_test.tiff", cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2RGB)
# Images are saved in 12 bit raw -> shift 4 bits
img = np.right_shift(img, 4)
img = img.astype(np.uint8)
img_pil = Image.fromarray(img)
#cv2.imshow("a",img)
#cv2.waitKey(1)
#plt.imshow(img_pil)
#plt.show()

#img = Image.open("data_test.tiff")
input_size = 300
img_pil = img_pil.resize((input_size,input_size)) 
#print(len(inputs[0][0]))
#print(len(img))
#inputs = 
#tensor = utils.prepare_tensor(inputs, precision == 'fp16')
#print(uris[0])
#print(utils.prepare_input(uris[0]))
a = preprocessFn(img_pil).unsqueeze(0)
print(a.shape)
#print(tensor.shape)
#print(img.mode)

a = a.to(device)
ssd300 = SSD300(backbone=ResNet('resnet18', None))

ssd300.to(device)

#Training

ssd300.train()
dboxes = dboxes300_coco()

#trainLoader
#valLoader
criterion = Loss(dboxes)
criterion.to(device)

learningRate = 0.05
optimizer = optim.SGD(tencent_trick(ssd300) lr = learningRate) #Tencent_trick is to disable weight decay for some parameters
epoch_size = 10
for epoch in range(0,epoch_size):
	#train_loop() #Put all paramters in here

	#Validation step

	#print out Accuracies and Losses







##Testing an image

ssd300.eval()
with torch.no_grad():
	yhat = ssd300(a)
print(len(yhat))
print(yhat[0].shape)
print(yhat[1].shape)
results_per_input = utils.decode_results(yhat)
best_results_per_input = [utils.pick_best(results, 0.50) for results in results_per_input]
classes_to_labels = {
  0: "red",
  1: "yellow",
  2: "green"
}
for image_idx in range(len(best_results_per_input)):
    fig, ax = plt.subplots(1)
    # Show original, denormalized image...
    image = img_pil#inputs[image_idx] / 2 + 0.5
    ax.imshow(image)
    # ...with detections
    bboxes, classes, confidences = best_results_per_input[image_idx]
    for idx in range(len(bboxes)):
        left, bot, right, top = bboxes[idx]
        x, y, w, h = [val * input_size for val in [left, bot, right - left, top - bot]]
        if(h>w):
	        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
	        ax.add_patch(rect)
	        ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
plt.show()

