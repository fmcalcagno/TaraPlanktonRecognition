import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")


def load_obj(name ):
    with open('../obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def myLoss(y_pred,y):
    return F.cross_entropy(y_pred, y.long())    


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

def ucf_collate(batch):
    label = torch.zeros(len(batch))
    input1 = torch.zeros(len(batch),3, 16, 112, 112)
    for i in range(len(batch)):
        input_label = batch[i]
        label[i] = int(input_label[1])
        input_list = input_label[0]
        for j in range(len(input_list)):
            input1[i][j] = input_list[j]
    return (input1, label)
