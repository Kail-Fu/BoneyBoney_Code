from matplotlib import pyplot as plt

def visualize_loss(loss, is_training): 
    x = [i for i in range(len(loss))]
    plt.plot(x, loss)
    if is_training:
        plt.title('Loss per batch')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.savefig("training_loss_plot.png")
    else:
        plt.title('Loss per epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig("testing_loss_plot.png")
    plt.clf()
    

def visualize_accuracy(accuracy, is_training): 
    x = [i for i in range(len(accuracy))]
    plt.plot(x, accuracy)
    if is_training:        
        plt.title('Accuracy per batch')
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.savefig("training_accuracy_plot.png")
    else:
        plt.title('Accuracy per epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig("testing_accuracy_plot.png")
    plt.clf()
    