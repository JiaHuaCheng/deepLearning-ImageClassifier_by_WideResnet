### YOUR CODE HERE
import torch
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, load_testing_images
from Configure import model_configs, training_configs, preprocess_configs

parser = argparse.ArgumentParser()
parser.add_argument("mode", help="train, test, predict")
parser.add_argument("data_path", help="path to load CIFAR-10 and private dataset")
parser.add_argument("--checkpoint", help="path to save training checkpoint file")
parser.add_argument("--save_dir", help="path to save the final prediction")
args = parser.parse_args()

if __name__ == '__main__':

	# fix random seed for reproducing result.
	torch.manual_seed(0)
	torch.cuda.manual_seed(0)

	# load data ->  initilize model -> use data to train -> use public data to test.
	if args.mode == 'train':
		print("This part is for training.")

		model = MyModel(model_configs, training_configs) # initilize model by passing preset configs (saved under Configure.py)		
		training_loader, public_testing_loader = load_data(args.data_path, preprocess_configs)
		model.train(training_loader, training_configs, public_testing_loader)
		model.evaluate(public_testing_loader)
		print("Training done.")


	# Testing on public testing dataset
	elif args.mode == 'test':
		print("This part is for testing on public dataset.")

		model = MyModel(model_configs, args.checkpoint)
		_, public_testing_loader = load_data(args.data_path, preprocess_configs)
		public_test_accuracy = model.evaluate(public_testing_loader)
		print("Accuracy on public test datasets is : ", public_test_accuracy)
	

	# Predicting and storing results on private testing dataset 
	elif args.mode == 'predict':
		print("This part is for testing on private dataset.")
		
		private_testing_loader = load_testing_images(args.data_path, preprocess_configs)
		model = MyModel(model_configs, args.checkpoint)
		predictions = model.predict_prob(private_testing_loader)
		private_test_results_path = os.path.join(args.save_dir, "predictions")
		np.save(private_test_results_path, predictions) # save private prediction.npy
		print("Prediction saved.")
		

### END CODE HERE

