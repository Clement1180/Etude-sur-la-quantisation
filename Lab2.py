import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models

class AffineQuantizer:
    def __init__(self, q_min=-128, q_max = 127, epsilon = 1e-5):
        self.epsilon = epsilon
        self.q_min = q_min
        self.q_max = q_max
        self.scale = None
        self.z = None

    def Calibration(self,x):
        self.scale = (np.max(x)-np.min(x))/(self.q_max - self.q_min + self.epsilon)
        self.z = self.q_min - np.min(x)/self.scale
        self.z = np.clip(np.round(self.z), self.q_min, self.q_max)

    def Calibration_percentile(self,x):
        x_max = np.percentile(x,95)
        x_min = np.percentile(x,5.0)
        self.scale = (x_max-x_min)/(self.q_max - self.q_min + self.epsilon)
        self.z = self.q_min - x_min/self.scale
        self.z = np.clip(np.round(self.z), self.q_min, self.q_max)


    def Quantize(self,x):
        q = np.round(x/self.scale + self.z)
        q = np.clip(q, self.q_min, self.q_max)
        return q
    def Dequantize(self,q):
        x = self.scale*(q - self.z)
        return x


    def forward(self,x):
        self.Calibration(x)
        q = self.Quantize(x)
        x_rec = self.Dequantize(q)
        return x_rec
    
    def forward_percentile(self,x):
        self.Calibration_percentile(x)
        q = self.Quantize(x)
        x_rec = self.Dequantize(q)
        return x_rec
    

x_normal = np.random.randn(1000).astype(np.float32)
x = np.append(x_normal, [100.0]).astype(np.float32)
quantiser = AffineQuantizer()
x_rec = quantiser.forward(x)
MSE = 1/len(x)*(np.sum((x-x_rec)**2))

print (MSE)


''' test sur un resnet'''

model = models.resnet18(weights = "DEFAULT")
model.eval()

weights_list = []
for param in model.parameters():
    weights_list.append(param.detach().cpu().numpy().flatten())
real_weights = np.concatenate(weights_list)

print(f"Statistiques des poids réels :")
print(f"Min: {real_weights.min():.4f}, Max: {real_weights.max():.4f}")
print(f"Moyenne: {real_weights.mean():.4f}, Std: {real_weights.std():.4f}")


x_rec = quantiser.forward(real_weights)
MSE = np.mean((real_weights - x_rec)**2)
print(f"MSE sur les poids du ResNet sans utilisation de la distribution: {MSE}")


''' test sur resnet en utilisant la distribution pour éliminer les outlayers'''

weights_list = []
for param in model.parameters():
    weights_list.append(param.detach().cpu().numpy().flatten())
real_weights = np.concatenate(weights_list)




x_rec = quantiser.forward_percentile(real_weights)
MSE = np.mean((real_weights - x_rec)**2)
print(f"MSE sur les poids du ResNet avec utilisation de la distribution: {MSE}")