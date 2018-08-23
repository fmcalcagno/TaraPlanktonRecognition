"""
PyTorch Panckton Tara Project
This project takes a dataset of more than 100k  Gif and creates a deep learning model 
using a modified C3D architecture to recognize from 4 different types of Plankton hierarchies

Facundo Calcagno 2018

Usage:
    Start tensorboard:
    ```bash
    tensorboard --logdir=/tmp/tensorboard_logs/
    ```
    Run the example:
    ```bash
    python taraPlankton.py --log_dir=/tmp/tensorboard_logs
    ```
"""

from __future__ import print_function
from argparse import ArgumentParser
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn


import InputInformation as Input
from C3DModel import C3D
import Dataloaders as planctonDataLoaders
import UtilsPL

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy, Loss, Recall, Precision
from ignite.metrics.metric import Metric
from MyAccuracies import myCategoricalAccuracy,myRecall,myPrecision


import warnings

def run(train_batch_size, val_batch_size, epochs, lr, momentum, log_interval, log_dir,classif):
	
	use_cuda =  torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	
	pwargs = {'rootDir': "../taraImages/",  'channels':1, 'timeDepth':16,
             'xSize':112, 'ySize':112, 'startFrame':0, 'endFrame':15,'numFilters':3,'filters':[2,3,4], 
             'transfpos':2}
	kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
		
	batchargs = {'train_batch_size':train_batch_size,'val_batch_size':val_batch_size }

	#log_dir="../logs/"
	
	trainset="../plancton-train.csv"
	testset="../plancton-test.csv"
	#Type of classification:   2, 13 or 156 classes
	transfpos=classif
	#epochs=1
	#log_interval=10000

	trainlist,testlist,hierclasses,classes_transf,outputsize=Input.inputInformation(trainset,testset,transfpos)
	print("Starting training for  # Images: {} epochs: {}, batch size: {}, lr: {:.4f}, Output Classes: {}, using: {}"
			  .format(len(trainlist), epochs, train_batch_size,lr,outputsize,device ))
	c3d = C3D(outputsize)
	c3d.apply(UtilsPL.weights_init)
	if use_cuda:
		c3d.cuda()
		c3d = nn.DataParallel(c3d, device_ids=range(torch.cuda.device_count()))
		cudnn.benchmark = True	
	optimizer = torch.optim.Adam(c3d.parameters(), lr=lr,weight_decay=0.0001)
	
	model=c3d
	train_loader, val_loader = planctonDataLoaders.get_data_loaders(trainlist,testlist,hierclasses,classes_transf,pwargs,kwargs,**batchargs)
	writer = UtilsPL.create_summary_writer(model, train_loader, log_dir)
	trainer = create_supervised_trainer(model, optimizer,UtilsPL.myLoss, device=device)
	
	evaluator = create_supervised_evaluator(model,
                                        metrics={'accuracy': myCategoricalAccuracy(),
                                                 'crossEntropyLoss': Loss(UtilsPL.myLoss),
                                                 'recall': myRecall(True),
                                                 'precision': myPrecision(True)},
                                        device=device)
	
	@trainer.on(Events.ITERATION_COMPLETED)
	def log_training_loss(engine):
		iter = (engine.state.iteration - 1) % len(train_loader) + 1
		if iter % log_interval == 0:
			print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
				  "".format(engine.state.epoch, iter, len(train_loader), engine.state.output))
			writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

	@trainer.on(Events.EPOCH_COMPLETED)
	def log_training_results(engine):
		evaluator.run(train_loader)
		metrics = evaluator.state.metrics
		avg_accuracy = metrics['accuracy']
		avg_cel = metrics['crossEntropyLoss']
		print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
			  .format(engine.state.epoch, avg_accuracy, avg_cel))
		writer.add_scalar("training/avg_loss", avg_cel, engine.state.epoch)
		writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)

	@trainer.on(Events.EPOCH_COMPLETED)
	def log_validation_results(engine):
		evaluator.run(val_loader)
		metrics = evaluator.state.metrics
		avg_accuracy = metrics['accuracy']
		avg_cel = metrics['crossEntropyLoss']
		avg_recall = metrics['recall']
		avg_precision = metrics['precision']
		print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f} Recall {:.2f} Precision {:.2f}"
			  .format(engine.state.epoch, avg_accuracy, avg_cel, avg_recall,avg_precision))
		writer.add_scalar("valdation/avg_loss", avg_cel, engine.state.epoch)
		writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.epoch)
		writer.add_scalar("valdation/recall", avg_recall, engine.state.epoch)
		writer.add_scalar("valdation/precision", avg_precision, engine.state.epoch)
		
	trainer.run(train_loader, max_epochs=epochs)
	writer.close()

		
	

if __name__ == "__main__":
	
	warnings.filterwarnings("ignore")
	
	parser = ArgumentParser()
	
	parser.add_argument('--batch_size', type=int, default=15,
                        help='input batch size for training (default: 15)')
	parser.add_argument('--val_batch_size', type=int, default=15,
                        help='input batch size for validation (default: 15)')
	parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
	parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
	parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
	parser.add_argument("--log_dir", type=str, default="../logs/",
                        help="log directory for Tensorboard log output")
	parser.add_argument('--classif', type=int, default=2,
                        help='Type of Classification (default:2 (156 Classes), 0:(2 Classes)')
	

	args = parser.parse_args()

	run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum,
        args.log_interval, args.log_dir,args.classif)
