transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
     transforms.Normalize((0.4376, 0.4437, 0.4727), (0.1980, 0.2010, 0.1969))])

transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
     transforms.Normalize((0.4524, 0.4525, 0.4690), (0.2194, 0.2266, 0.2285))])

trainset = torchvision.datasets.SVHN(root='./data', split = 'train',  transform=transform_train,
                                        download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.SVHN(root='./data', split = 'test',  transform=transform_test, 
                                       download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
