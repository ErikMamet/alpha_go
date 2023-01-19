import torch 
import numpy as np
import os 
import os.path as osp
import go_cnn
import dataset_manager as data_m

###
# Data & hyperparams
###
SIZE_OF_INPUT = 5+1 

net = go_cnn.GoCNN(SIZE_OF_INPUT )
dataset = data_m.Go9x9_Dataset(data_dir="./data/mini_dataset")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader =  torch.utils.data.DataLoader(dataset, batch_size=32, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, drop_last=True)


###
# Utils
###
def resume_training(root = "./log"):
    i = 0
    path = osp.join(root, "checkpoint_epoch_"+str(i))
    while osp.exists(path):
        i+= 1
        path = osp.join(root, "checkpoint_epoch_"+str(i))

    return i, path

def save(model, epoch, root = "./log"):
    path = osp.join(root, "checkpoint_epoch_"+str(epoch))
    torch.save(model.state_dict(), path)

def log(epoch, test_accuracy, loss, root = "./log", testing_accuracy = -1 ):
    if not osp.exists(osp.join(root, "log.txt")):
        f = open(osp.join(root, "log.txt"), mode = 'a')
        f.writelines("New training : \n")
    else :
        f = open(osp.join(root, "log.txt"), mode = 'a')
    f.writelines("for epoch "+str(epoch)+" ; avg training acc : ", str(test_accuracy),  "; avg training loss : ", loss , "; testing accuracy : ", str(testing_accuracy) ,"\n")


def calc_acc(outputs, labels):
    acc = 0
    for i in range(labels.size()[0]):
        acc += (torch.argmax(labels) == torch.argmax(outputs))
    return "accuracy calculator not implemented yet"

###
# testing and training
###

def testing(model, dataloader):
    if torch.cuda.is_available():
        device = "cuda"
    else :
        device = "cpu"
    
    avg_test_acc = 0
    l = 0
    for elem, lab in dataloader :
        torch.no_grad()
        #passing batch data to gpu
        elem = elem.to(device)
        lab = lab.to(device)
        value, policy = model(elem)
        outputs = value*policy
        test_acc = calc_acc(outputs, lab)
        avg_test_acc += test_acc
        l+= 1
    avg_test_acc = avg_test_acc / l #ceci fonctionne car l'accuracy est correct/batch_size 
    return avg_test_acc

def training(model, max_epochs, lr , dataloader, testing_freq = 5):

    if torch.cuda.is_available():
        device = "cuda"
    else :
        device = "cpu"
    
    #loss function 
    criterion = torch.nn.BCELoss()

    #optimiser and schedueler(update hyper params during training, mostly decrease learning rate)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    #schedueler = torch.optim.lr_scheduler.StepLR(optimizer=optimiser)

    #passing model to gpu
    model = model.to(device)

    #resuming training from checkpoint if possible
    start, checkpoint = resume_training() 
    if start != 0 :
        model.load_state_dict(torch.load(checkpoint))


    for _ in range(start, max_epochs):
        
        avg_test_acc = 0
        avg_train_loss = 0
        l = 0
        for elem, lab in dataloader :
            print("elem size ", elem.size())
            optimiser.zero_grad()

            #passing batch data to gpu
            elem = elem.to(device)
            lab = lab.to(device)

            value, policy = model(elem)
            outputs = value*policy
            loss = criterion(outputs, lab)
            loss.backward()
            optimiser.step()

            test_acc = calc_acc(outputs, lab)
            avg_test_acc += test_acc
            print("for epoch " + str(_) + "& batch " , + str(l) + " ; acc : " +test_acc, ", loss : ", str(loss))
            l+= 1
        avg_test_acc = avg_test_acc / l
        avg_train_loss = avg_train_loss / l

        #save model
        #also implement function to log 
        save(model=model, epoch= _)
        log(_, test_accuracy=avg_test_acc, loss=avg_train_loss)
        if _ % testing_freq == 0:
            testing(model, train_loader)
        #test model
            


if __name__ == "__main__":
    model = net
    training(model=model, max_epochs=5, lr= 0.1, dataloader=train_loader)
