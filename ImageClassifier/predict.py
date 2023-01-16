import argparse
import process_img # import the module containing the image preprocessing functions
import vgg_arch # import the module containing the model architecture and loading functions
import torch
import json


def get_args():
    """
    Define and parse command line arguments for the script.
    
    Returns:
        Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Make predictions using a pre-trained model")
    parser.add_argument("input_image", type=str, help="path to the input image")
    parser.add_argument("checkpoint_vgg", type=str, help="path to the checkpoint file")
    parser.add_argument("--top_k", type=int, default=5, help="number of top predictions to return")
    parser.add_argument("--category_names", type=str, default='/home/workspace/ImageClassifier/cat_to_name.json', help="path to the JSON file containing the label to category mapping")
    parser.add_argument("--use_gpu", action='store_true', default=False, help="use GPU for inference")
    return parser.parse_args()



def predict(image, model, topk, category_names, use_gpu, custom_class_mapping=None):
    """
    Make a prediction for an input image
    Args:
    image: PIL image
    model: trained Pytorch model
    topk: number of top predictions to return
    category_names: path to the JSON file containing the label to category mapping
    use_gpu: whether to use GPU for inference
    custom_class_mapping: a custom dictionary to map class indices to class names, optional
    Returns:
    probs: list of probabilities of the topk predictions
    classes: list of class names of the topk predictions
    """
    model.eval() # set model to evaluation mode
    class_to_idx = model.class_to_idx
    idx_to_class = {class_to_idx[k]: k for k in class_to_idx} # create a mapping from class index to class name

    if custom_class_mapping:
        idx_to_class = custom_class_mapping # use the custom class mapping instead if provided
    
    image = torch.from_numpy(image.numpy()).float() # convert the image to a tensor
    image = torch.unsqueeze(image, dim=0) # add a batch dimension

    if use_gpu and torch.cuda.is_available():
        print("Using GPU")
        image = image.cuda()
        model = model.cuda()

    with torch.no_grad():
        output = model.forward(image)
        preds = torch.exp(output).topk(topk)
    probs = preds[0][0].cpu().data.numpy().tolist() # convert the probabilities to a list
    classes = preds[1][0].cpu().data.numpy() # get the class indices
    classes = [idx_to_class[i] for i in classes] # convert class indices to class names
    if category_names != '':
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[x] for x in classes] # convert class names to actual names using the provided JSON file

    return probs, classes





def main():
    args = get_args()
    processed_img = process_img.process_image(args.input_image) # preprocess the image
    model = vgg_arch.load_model(args.checkpoint_vgg) # load the model from checkpoint
    probs, classes = predict(processed_img, model, args.top_k, args.category_names, args.use_gpu)
    print(f"Top {args.top_k} predictions: {list(zip(classes, probs))}")

if __name__ == '__main__':
    main()
