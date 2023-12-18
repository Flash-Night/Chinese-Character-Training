import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import os
from PIL import Image 

class Encoder(nn.Module):
    
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 15 * 15, 256)
        self.FC_mean  = nn.Linear(256, latent_dim)
        self.FC_var   = nn.Linear(256, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.1)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.conv1(x))
        h_       = self.LeakyReLU(self.conv2(h_))
        h_       = self.LeakyReLU(self.conv3(h_))
        h_       = h_.view(-1, 64 * 15 * 15)
        h_       = self.LeakyReLU(self.fc1(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)

        return mean, log_var
    
    
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 64 * 15 * 15)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        
        self.LeakyReLU = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.fc1(x))
        h     = self.LeakyReLU(self.fc2(h))
        h     = h.view(-1, 64, 15, 15)
        h     = self.LeakyReLU(self.deconv1(h))
        h     = self.LeakyReLU(self.deconv2(h))
        h     = torch.sigmoid(self.deconv3(h))
        return h



class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var
    
       

def loss_function(x, x_hat, mean, log_var):
    #reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


'''
This function is not really necessary. 
'''

def encoding( x, model):
    mean, log_var = model.Encoder(x)
    z = model.reparameterization(mean, torch.exp(0.5 * log_var))
    return z


class ImageDataset(torch.utils.data.Dataset): 
    
    def __init__(self, image_dir, RGB = False, transform=None): 
        self.data_dir = image_dir
        # self.images = os.listdir(image_dir) 
        self.images=[img for img in os.listdir(image_dir) if img.endswith(('png','jpg','jpeg'))]
        self.transform = transform 
        self.RGB = RGB

    # Defining the length of the dataset 
    def __len__(self): 
        return len(self.images) 
  
    # Defining the procedure to obtain one item from the dataset 
    def __getitem__(self, index): 
        image_path = os.path.join(self.data_dir, self.images[index]) 
        
        image = Image.open(image_path)
        
        if self.RGB: 
            image = image.convert("RGB")
        else: 
            image = image.convert("L")
        
        image = np.array(image).astype(np.float32)
        # image = np.where(image<16, 0, 1)
        image = image/255
        
        #print("image from Pil", image.shape) #H, W, channels
  
        # Applying the transform 
        if self.transform: 
            image = self.transform(image) 
        
        #print("Image after transform", image.shape) #chanels, H< W
        return image



if __name__ == "__main__":
    
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    
    from torch.utils.data import DataLoader
    
    from torch.optim import Adam
    
    
    DEVICE = torch.device("cuda" 
                          if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available()  
                          else "cpu")
    print("device:", DEVICE)
      
    
    size = (120, 120) 
    
    latent_dim = 40

    data_path="C:\Study-cityu\y1-semA\machine learning\hanzi\kaishu"
    pt = "C:\Study-cityu\y1-semA\machine learning\hanzi\model.pt"
    decoder_pt = "C:\Study-cityu\y1-semA\machine learning\hanzi\model_decoder.pt"

    '''
    instantiate the model 
    '''
    
    encoder = Encoder(latent_dim)
    decoder = Decoder(latent_dim)
    
    model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
    model.load_state_dict(torch.load(pt))
    
    '''
    Dataset processing
    '''

    
    batch_size = 100
    learning_rate = 1e-3
    epochs = 500
    
    #transformer 
    mnist_transform = transforms.Compose([
            transforms.ToTensor(),
    ])
    
    #dataloader
    dataset = ImageDataset(data_path)
    
    loader = DataLoader(dataset=dataset, 
        batch_size=batch_size, 
        # If true, shuffles the dataset at every epoch 
        shuffle=True
    )
    
    
    '''
    Optimizer
    '''
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    
    '''
    Training
    
    '''
    
    model.train()
    
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, x in enumerate(loader):
            
            x = x.unsqueeze(0)
            x = x.transpose(0,1)
            # print(x.shape)
            # x = x.view(x.shape[0], 120, 120)
            x = x.to(DEVICE)
    
            optimizer.zero_grad()
    
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
        
    print("Finished!!")
    
    
    
    #we finished training so we set model to eval( )
    model.eval()
    torch.save(model.state_dict(), pt)
    torch.save(model.Decoder.state_dict(), decoder_pt)
    
    from PIL import Image #not used in this version
    
    import matplotlib.pyplot as plt
    
    with torch.no_grad():
        
        #generate a batch of fake images 
        noise = torch.randn(batch_size, latent_dim).to(DEVICE)
        generated_images = model.Decoder(noise)
        
        
        #show two generated images
        for idx in range(2): 
            #assume grayscale image... 
            x = generated_images.view( batch_size, size[0], size[1])[idx].cpu()
            x = x.squeeze()
            
            x = x.numpy() 
            
            #change the range from (0,1) to (0,255)
            x = (x * 255)
            # x = np.where(x < 16, 0, 255)
            #convert to int datatype 
            x = x.astype(np.uint8)
            print(x)
            
            plt.figure()
            plt.imshow(x, interpolation='none', cmap="gray") 
            plt.show()
            
