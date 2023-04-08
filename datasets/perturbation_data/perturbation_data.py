class GaussianNoise:
    def __init__(self,train_loader,device,sigma=0.05):
        self.train_loader = train_loader
        self.device = device
        self.sigma = sigma

    def get_noise_data(self):
        data_iterator = iter(self.train_loader)
        input , label = next(data_iterator)
        x, target = (input), label.float()
        x, target = x.to(self.device), target.to(self.device)

        data = x.to(self.device)
        target = target.to(self.device)
        noise = (data.new(data.size()).normal_(0,self.sigma)).to(self.device)
        return data,target,noise
