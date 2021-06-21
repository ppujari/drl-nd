import os
import sys
import time
import json
import torch
import argparse
import numpy as np
from torch import nn
from PIL import Image
from torch import optim
from pathlib import Path
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets, transforms, models

import utils

def main(argv):
    """ Main Entry Point

        Arguments:
        argv -- system command line arguments
    """

    path_to_img = None
    path_to_checkpoint = str(Path(os.getcwd())) + str(Path('/checkpoint_train.pth'))
    top_k = 1
    path_to_category_names = str(Path(os.getcwd())) + str(Path('/cat_to_name.json'))
    use_gpu = False
    use_cat_to_name = False

    try:
        print("Flower Classifier Model Predictor based on VGGx Pre-Trained Neural Network - v1.0")

        parser = argparse.ArgumentParser()
        parser.add_argument("path_to_img_and_checkpoint", nargs=2)
        parser.add_argument("--top_k", "-tp", type=int, help="set top k")
        parser.add_argument("--path_to_category_names", "-ptcn", help="set category to real names json file")
        parser.add_argument("--gpu", "-g", action='store_true', help="set if to use GPU")
        args = parser.parse_args()

        if args.path_to_img_and_checkpoint:
            path_to_img = str(Path(str(args.path_to_img_and_checkpoint[0])))
            path_to_checkpoint = str(Path(str(args.path_to_img_and_checkpoint[1])))
            print("Argument 'path_to_img' = {}".format(path_to_img))
            print("Argument 'path_to_checkpoint' = {}".format(path_to_checkpoint))
        if args.top_k:
            top_k = args.top_k
            print("Argument 'top_k' = {}".format(top_k))
        if args.path_to_category_names:
            path_to_category_names = str(Path(args.path_to_category_names))
            use_cat_to_name = True
            print("Argument 'path_to_category_names' = {}".format(path_to_category_names))
        if args.gpu:
            use_gpu = args.gpu
            print("Argument 'gpu' = {}".format(use_gpu))
    except:
      print("Failed to parse command line arguments.")
      print("Unexpected error: {}".format(sys.exc_info()[0]))
      sys.exit(1)

    try:
        if use_cat_to_name == True:
            with open(path_to_category_names, 'r') as f:
                cat_to_name = json.load(f)

        if use_gpu == True:
            device = utils.get_fastest_device()
        else:
            device = utils.get_slowest_device()

        model = utils.create_model(102, utils.get_class_to_idx(path_to_checkpoint), "vgg11", 2048)
        criterion = nn.NLLLoss()
        # train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

        if torch.cuda.is_available():
            model = model.to(device)

        print("Starting Prediction...")
        print("Loading Model from: {}".format(path_to_checkpoint))
        epochs, optimizer, model = utils.load_checkpoint(path_to_checkpoint, optimizer, model)
        print("Performing Prediction...")
        image, probs, classes = predict(path_to_img, device, model, top_k)
        print("Class Probability(s): {}".format(probs))
        print("Class Index(es): {}".format(classes))
        if use_cat_to_name == True:
            flowers = flower_predictions(classes, cat_to_name)
            print("Class Flower(s): {}".format(flowers))
        print("Success!!!")
    except:
        print("Failed to initialize for model prediction.")
        print("Unexpected error: {}".format(sys.exc_info()[0]))
        sys.exit(1)

def predict(image_path, device, model, topk=5):
    """ Predict Image Class

        Arguments:
        image_path -- path to image to classify
        device -- fastest supported device (cpu or cuda)
        model -- nerual net model to train
        topk -- top number of probabilities to return
    """
    try:
        image_ = utils.process_image(image_path)
        image_ = image_.unsqueeze(0)
        image_ = image_.to(device)
        model.eval()
        with torch.no_grad():
            output_ = model.forward(image_)
        probs_, classes_ = torch.exp(output_).topk(topk)
        probs_ = probs_.cpu().detach().numpy().flatten()
        classes_ = classes_.cpu().detach().numpy().flatten()
        model_idx_to_class_ = {value_: key_ for key_, value_ in model.class_to_idx.items()}
        classes_ = [model_idx_to_class_[class_] for class_ in classes_]
    except:
        print("Failed to predict image.")
        print("Unexpected error: {}".format(sys.exc_info()[0]))
        sys.exit(1)
    
    return image_, probs_, classes_

def flower_predictions(classes, class_dict):
    """ Predict Flower

        Arguments:
        classes -- flower class indexes
        class_dict -- class to flower dictionary
    """

    flowers_ = [class_dict[class_] for class_ in classes]

    return flowers_

if __name__ == "__main__":
   main(sys.argv[1:])