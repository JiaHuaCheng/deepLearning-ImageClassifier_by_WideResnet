package: numpy, torchvision, PIL, torch, tqdm, 

Setup: 1. follow google drive link under saved_model folder to download model parameter.
       2. put download file final_epoch200.pt under checkpoint folder.
       3. Download whole code folder into local machine.
       4. Before execute code, check our code folder has all the following files.
       
		1 checkpoint folder and has parameter file "final_epoch200.pt" in it.
		1 results folder 
		1 data folder. Private_test_dataset should be put in this folder. 
		6 python files(main.py, Model.py, Network.py, ImageUtils.py, Configure.py, Dataloader.py)

      5. Execute code like below

	For training,
	python main.py train data
	For testing,
	python main.py test data --checkpoint ./checkpoint/final_epoch200.pt 
	For prediction: 
	python main.py predict data --checkpoint ./checkpoint/final_epoch200.pt --save_dir ./results
