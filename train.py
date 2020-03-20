import argparse
import torch
import torch.nn as nn
import torchvision
import numpy as np
import os
import model
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

#custom imports
import load_data as ld
from load_data import PhoneDataset

#global variables related to the image dataset properties
IMG_WIDTH = 490
IMG_HEIGHT = 326

def main(args):
    #device configuration
    if args.cpu != None:
        device = torch.device('cpu')
    elif args.gpu != None:
        if not torch.cuda_is_available():
            print("GPU / cuda reported as unavailable to torch")
            exit(0)
        device = torch.device('cuda')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model directory
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    train_data = ld.get_data(labels_file=args.labels_file,
                           root_dir=args.train_image_dir, mode="absolute");

    validation_data = ld.get_data(labels_file=args.labels_file,
                                   root_dir=args.validation_image_dir, mode="absolute");

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True);
    val_loader = DataLoader(dataset=validation_data, batch_size=args.validation_batch_size)


    # Build the models
    if args.num_layers != None and args.block_type != None:
        if args.block_type == "bottleneck":
            net = model.ResNet(model.Bottleneck, args.num_layers, dropout=args.dropout)
        else:
            net = model.ResNet(model.BasicBlock, args.num_layers, dropout=args.dropout)
    else:
        if args.resnet_model == 152:
            net = model.ResNet152(args.dropout)
        elif args.resnet_model == 101:
            net = model.ResNet101(args.dropout)
        elif args.resnet_model == 50:
            net = model.ResNet50(args.dropout)
        elif args.resnet_model == 34:
            net = model.ResNet34(args.dropout)
        else:
            net = model.ResNet101(args.dropout)

    #load the model to the appropriate device
    net = net.to(device)
    params = net.parameters()

    # Loss and optimizer
    criterion = nn.MSELoss() #best for regression

    if args.optim != None:
        if args.optim == "adadelta":
            optimizer = torch.optim.Adadelta(params, lr=args.learning_rate)
        if args.optim == "adagrad":
            optimizer = torch.optim.Adagrad(params, lr=args.learning_rate)
        if args.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=args.learning_rate)
        if args.optim == "adamw":
            optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
        if args.optim == "rmsprop":
            optimizer = torch.optim.RMSProp(params, lr=args.learning_rate)
        if args.optim == "sgd":
            optimizer = torch.optim.SGD(params, lr=args.learning_rate)
    else:
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)


    val_acc_history = []
    train_acc_history = []
    failed_runs = 0
    prev_loss = float("inf")

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        total_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader, 0):
            net.train()

            #adjust to output image coordinates
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs.float())
            loss = criterion(outputs.float(), labels.float())
            loss.backward()


            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clipping_value)
            optimizer.step()
            running_loss += loss.item()
            total_loss += loss.item()
            if i % 2 == 0: #print every mini-batches
              print('[%d, %5d] loss: %.5f' %
                    (epoch + 1, i + 1, running_loss/2))
              running_loss = 0.0

        loss = 0.0

        #compute validation loss at the end of the epoch
        for i, (inputs, labels) in enumerate(val_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            net.eval()
            with torch.no_grad():
                outputs = net(inputs.float())
                loss += criterion(outputs, labels.float()).item()

        print("------------------------------------------------------------")
        print("Epoch %5d" % (epoch+1))
        print("Training loss: {}, Avg Loss: {}".format(total_loss, total_loss / train_data.__len__()))
        print("Validation Loss: {}, Avg Loss: {}".format(loss, loss / validation_data.__len__()))
        print("------------------------------------------------------------")

        val_acc_history.append(loss)
        train_acc_history.append(total_loss)

        #save the model at the desired step
        if (epoch+1) % args.save_step == 0:
          torch.save(net.state_dict(), args.model_save_dir+"resnet"+str(epoch+1)+".pt")

        ##stopping conditions
        if failed_runs > 5 and prev_loss < loss:
          break
        elif prev_loss < loss:
          failed_runs += 1
        else:
          failed_runs = 0

        prev_loss = loss


    #create a plot of the loss
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1, len(val_acc_history)+1),val_acc_history,label="Validation loss")
    plt.plot(range(1,len(train_acc_history)+1), train_acc_history,label="Training loss")
    plt.xticks(np.arange(1, len(train_acc_history)+1, 1.0))
    plt.legend()
    plt.ylim((0, max([max(val_acc_history), max(train_acc_history)])))

    if args.save_training_plot != None:
        plt.savefig(args.save_training_plot+"loss_plot.png")

    plt.show()
    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_image_dir', type=str, default='./data/train/', help='path to training set')
    parser.add_argument('--validation_image_dir', type=str, default='./data/validation/', help='path to validation set')
    parser.add_argument('--labels_file', type=str, default='./data/labels/labels.txt', help='path to labels file for both validation and training datasets')
    parser.add_argument('--model_save_dir', type=str, default='./saved_models/', help='path to location to save models')
    parser.add_argument('--save_step', type=int , default=4, help='step size for saving trained models')
    parser.add_argument('--save_training_plot', nargs='?', type=str, const='./', help='location to save a plot showing testing and validation loss for the model')

    # Model parameters
    parser.add_argument('--optim', type=str, default="adam", help="options such as adagrad, adadelta, sgd, etc.")
    parser.add_argument('--block_type', type=str, default="bottleneck", help='type of resnet layer (bottleneck or basic)')
    parser.add_argument('--num_layers', type=int , nargs=4, help='input of four space-separated integers (i.e. 1 2 30 2 ) where each number represents the number of blocks at that respective layer')
    parser.add_argument('--resnet_model', type=int , nargs=1, default=101, help='use to specify a pre-designed resnet model (18 34 50 101 152). NOTE: this option will be overriden if both num_layers and block_type are specified')
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--validation_batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clipping_value', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--cpu', nargs='?', type=bool, const=True)
    parser.add_argument('--gpu', nargs='?', type=bool, const=True)
    args = parser.parse_args()

    print(args)
    main(args)
