import torch 
import numpy as np
import os 
import os.path as osp
import go_cnn
import dataset_manager as data_m
import time
from tqdm import tqdm

###
# Data & hyperparams
###
SIZE_OF_INPUT = 5+1 
#BATCH_SIZE = 32768
BATCH_SIZE = 8192
#BATCH_SIZE = 128

net = go_cnn.GoCNN(SIZE_OF_INPUT)
dataset = data_m.Go9x9_Dataset(data_dir="./data/dataset")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

print(train_dataset.__len__() / BATCH_SIZE)

train_loader =  torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, num_workers=12)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True, num_workers=12)


if torch.cuda.is_available():
    print('GPU has been found')
    device = "cuda"
else :
    device = "cpu"
###
# Utils
###
def resume_training(root = "./log"):
    i = 1
    path = osp.join(root, "checkpoint_epoch_"+str(i))
    path_last = osp.join(root, "checkpoint_epoch_"+str(0))

    while osp.exists(path):
        path_last = osp.join(root, "checkpoint_epoch_"+str(i))
        path = osp.join(root, "checkpoint_epoch_"+str(i+1))
        i+= 1

    return i-1, path_last

def save(model, epoch, root = "./log"):
    path = osp.join(root, "checkpoint_epoch_"+str(epoch+1))
    torch.save(model.state_dict(), path)

def log(epoch, train_accuracy, loss, test_accuracy, root = "./log"):
    if not osp.exists(osp.join(root, "log.txt")):
        f = open(osp.join(root, "log.txt"), mode = 'a')
        f.writelines("New training : \n")
    else :
        f = open(osp.join(root, "log.txt"), mode = 'a')
    f.writelines("for epoch "+str(epoch)+" ; avg training acc : "+ str(train_accuracy)+  "; avg training loss : "+ str(float(loss)) + "; testing accuracy : "+ str(test_accuracy) +"\n")


def calc_acc(policy, policy_target, value, value_target):
    acc_policy = 0
    acc_value = 0
    for i in range(policy.size()[0]):
        if (torch.argmax(policy_target[i]) == torch.argmax(policy[i])): acc_policy += 1
        if ((value[i] > 0 and value_target[i] == 1) or (value[i] < 0 and value_target[i] == -1)) : acc_value += 1 
    
    total_acc_policy = int(acc_policy)/policy_target.size()[0]
    total_acc_value = int(acc_value)/value_target.size()[0]
    return total_acc_policy, total_acc_value


###
# testing and training
###

def testing(model, dataloader):

    avg_test_acc_policy = 0
    avg_test_acc_value = 0
    l = 0
    for elem, lab in dataloader :
        torch.no_grad()
        #passing batch data to gpu
        elem = elem.to(device)
        policy_target = lab[0].to(device)
        value_target = lab[1].to(device)
        policy, value = model(elem)
        test_acc_policy, test_acc_value = calc_acc(policy, policy_target, value, value_target)
        avg_test_acc_policy += test_acc_policy
        avg_test_acc_value += test_acc_value
        l+= 1
    avg_test_acc_policy = avg_test_acc_policy / l #ceci fonctionne car l'accuracy est correct/batch_size 
    avg_test_acc_value = avg_test_acc_value / l
    return avg_test_acc_policy, avg_test_acc_value

def training(model, max_epochs, lr , train_dataloader, test_dataloader, testing_freq = 5):
    t = time.time()
    #loss function 
    criterion1 = torch.nn.BCELoss()
    criterion2 = torch.nn.MSELoss()

    #optimiser and scheduler(update hyper params during training, mostly decrease learning rate)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimiser)

    #resuming training from checkpoint if possible
    start, checkpoint = resume_training() 
    if start > 0 :
        model.load_state_dict(torch.load(checkpoint))

    model = model.to(device)
    
    for epoch in range(start, max_epochs):
        t = time.time()
        avg_train_acc_policy = 0
        avg_train_acc_value = 0
        avg_train_loss1 = 0
        avg_train_loss2 = 0
        avg_train_loss = 0
        l = 0
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for elem, lab in tepoch:
                tepoch.set_description(f'Epoch {epoch+1}')
                optimiser.zero_grad()

                #passing batch data to gpu
                elem = elem.to(device)
                policy_target = lab[0].type(torch.FloatTensor).to(device)
                value_target = lab[1].type(torch.FloatTensor).to(device)
                policy, value = model(elem)
                loss1 =  criterion1(policy, policy_target)
                loss2 = criterion2(value, value_target.view(-1,1))
            
                loss = loss1 + loss2 
                loss.backward()
                optimiser.step()

                acc_policy, acc_value = calc_acc(policy, policy_target, value, value_target)
                avg_train_acc_policy += acc_policy
                avg_train_acc_value += acc_value
                avg_train_loss1 += loss1
                avg_train_loss2 += loss2
                avg_train_loss += loss
                l+= 1
                tepoch.set_postfix(loss=f'{loss:.3f}', loss1=f'{loss1:.3f}', loss2=f'{loss2:.3f}', policyAcc=f'{acc_policy:.3f}', valueAcc=f'{acc_value:.3f}')
        avg_train_acc_policy = avg_train_acc_policy / l
        avg_train_acc_value = avg_train_acc_value / l
        avg_train_loss = avg_train_loss / l
        avg_train_loss1 = avg_train_loss1 / l
        avg_train_loss2 = avg_train_loss2 / l
        
        #save model
        #also implement function to log 
        save(model=model, epoch=epoch)
        #if epoch % testing_freq == 0:
        test_acc_policy, test_acc_value = testing(model, test_dataloader)
        log(epoch, train_accuracy = avg_train_acc_policy, test_accuracy=test_acc_policy, loss=avg_train_loss)
        print("test_policyAcc:", str(test_acc_policy), "| test_valueAcc", str(test_acc_value), " --- elapsed time", f'{(time.time() - t):.1f}')


if __name__ == "__main__":
    model = net
    print("training is starting")
    training(model=model, max_epochs=50, lr= 0.01, train_dataloader=train_loader, test_dataloader = test_loader)
