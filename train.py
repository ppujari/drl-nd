import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse

#parse keyword arguments 
def parser():
    ap = argparse.ArgumentParser(description='Parser of Train Script')
    # Add architecture selection to parser
    ap.add_argument('--arch', 
                        type=str, 
                        help='Choose architecture from torchvision.models')
    
    # Add checkpoint directory to parser
    ap.add_argument('--save_dir', 
                        type=str, 
                        help='Define save directory for checkpoints.')
    
    # Add GPU Option to parser
    ap.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU + Cuda for calculations')
    
    ap.add_argument('--hidden_units', 
                        type=int, 
                        help='Hidden units for DNN classifier')
    
    ap.add_argument('--learning_rate', 
                        type=float, 
                        help='Define gradient descent learning rate')
    ap.add_argument('--epochs', 
                        type=int, 
                        help='Number of epochs for training')
    # Parse args
    args = ap.parse_args()
    return args

def loader_model(architecture):
    # Load Defaults if none specified
    if architecture == None: 
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print("Network architecture specified as vgg16...")
    else: 
        exec("model = getattr(models, architecture)(pretrained=True))".format(architecture))
        model.name = architecture
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False 
        
    return model    
    
def check_gpu(gpu_arg):
   # If gpu_arg is false, return the cpu device
    if not gpu_arg:
        return torch.device("cpu")
    
    # else then check for CUDA before assigning it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead..")
    return device

def initial_classifier(model, hidden_units):
    if hidden_units == None: 
        hidden_units = 4096 #hyperparamters
        print("Number of Hidden Layers: 4096.")
    
    # Find input layers
    input_features = model.classifier[0].in_features
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier

#creat a function for validation, calcuate loss and accuracy
def validation(model, testloader, criterion):
    testloss = 0
    accuracy = 0
    device = 'cpu'
    for ii, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model.forward(inputs)
        testloss += criterion(outputs, labels).item()
        ps = torch.exp(outputs)
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()
    
    return testloss, accuracy


def trainer(model, trainloader, validloader, device, criterion, optimizer, epochs, print_every, steps):
    
    if type(epochs) == type(None):
        epochs = 5
        print("Number of epochs specificed as 5.")    
 
    print("Initializing Training......")
    
    for e in range(epochs):
        running_loss = 0
    
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
        
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
        
            #Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps%print_every ==0:
                model.eval()

                with torch.no_grad():
                    valid_loss, valid_accuracy = validation(model, validloader, criterion)
            
                valid_loss = valid_loss/len(validloader)
                valid_accuracy =valid_accuracy/len(validloader)
            
                print("Epoch: {}/{} | ".format(e+1, epochs),
                    "Training Loss: {:.3f} | ".format(running_loss/print_every),
                    "Validation Loss: {:.3f} | ".format(valid_loss),
                    "Validation Accuracy: {:.3f}% | ".format(valid_accuracy*100))
                    
                running_loss = 0

    return model

def initial_checkpoint(Model, Save_Dir, Train_data):
     
    if type(Save_Dir) == type(None):
        print("Directory is not specified, model will not be saved.")
    else:
        if isdir(Save_Dir):
            Model.class_to_idx = Train_data.class_to_idx
            
            # Create checkpoint dictionary
            checkpoint = {'arch': Model.name,
                          'classifier': Model.classifier,
                          'class_to_idx': Model.class_to_idx,
                          'state_dict': Model.state_dict()}
            
            # Save checkpoint
            torch.save(checkpoint, 'checkpoint.pth')

        else: 
            print("Directory is not found, model will not be saved.")




def model_validate(Model, Testloader, Device):
    correct = 0
    total = 0
    with torch.no_grad():
        Model.eval()
        for data in Testloader:
            images, labels = data
            images, labels = images.to(Device), labels.to(Device)
            outputs = Model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy achieved by the network on test images: %d%%' % (100 * correct / total))

def main():
    #get keyword arguments
    args = parser()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    
    Train_transforms =transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],                                                                  [0.229, 0.224, 0.225])])
    
    Test_transforms= transforms.Compose([transforms.RandomRotation(256),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],                                                                  [0.229, 0.224, 0.225])])
    
    Valid_transforms= transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],                                                                  [0.229, 0.224, 0.225])])
    
    
    Train_data=datasets.ImageFolder(data_dir+'/train',transform=Train_transforms)
    Test_data=datasets.ImageFolder(data_dir+'/test', transform=Test_transforms)
    Valid_data = datasets.ImageFolder(data_dir+'/valid', transform=Valid_transforms)

    Train_loader = torch.utils.data.DataLoader(Train_data,batch_size=64, shuffle=True)
    Test_loader = torch.utils.data.DataLoader(Test_data,batch_size=20)
    Valid_loader=torch.utils.data.DataLoader(Valid_data,batch_size=32)
    
    # Load Model
    model = loader_model(architecture=args.arch)
    model.classifier = initial_classifier(model, 
                                         hidden_units=args.hidden_units)
    
    
    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu)
    model.to(device)
    
    # Check for learning rate args
    if type(args.learning_rate) == type(None):
        learningrate = 0.001
        print("Learning rate specificed as 0.001")
    else: learningrate = args.learning_rate
    
    
    # initial criterion, optimizer, choose learning rate=0.001
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=learningrate)

    #epochs = 7
    print_every = 30
    steps = 0
    
    trained_model = trainer(model, Train_loader, Valid_loader, 
                                  device, criterion, optimizer, args.epochs, 
                                  print_every, steps)
                    
    print("Complete Training......")
    
    model_validate(trained_model, Test_loader, device)
    
    #save the checkpoint
    model.to('cpu')
    model.class_to_idx = Train_data.class_to_idx

    # Save the model
    initial_checkpoint(trained_model, args.save_dir, Train_data)
    
    
main()