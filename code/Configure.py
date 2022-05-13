# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE
'''
	We follow the conclusion of paper "Wide Residual Networks" (https://arxiv.org/abs/1605.07146)
	to train our model. In page 7 table4, the best Test error(%) on CIFAR-10 is depth = 28, k = 10
	In page10, author mentions (1). learning_rate = 0.1, (2). 200 epochs. (3). And refer to auther's github, batch_size is 128.
	
'''

preprocess_configs = {
	"crop_padding" : 4,
	"cutout_holes" : 1,
	"cutout_length" : 16, 
	"batch_size" : 128 
}

model_configs = {
	"name": 'MyModel',
	"save_dir": '../saved_models/',
	"depth": 28,
	"num_classes" : 10,
	"width" : 10,
	"drop_rate" : 0.3,
}

training_configs = {
	"epochs" : 200,
	"batch_size" : 128,
}

### END CODE HERE