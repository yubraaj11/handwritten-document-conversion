import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class IIITHWDataset(Dataset):
    '''
    Custom dataset for the IIIT-HW dataset.
    Loads image-text pairs for OCR tasks.
    
    Attributes:
        root_dir (str): Path to the root directory containing image files.
        df (DataFrame): A DataFrame containing file names and corresponding texts.
        processor (Processor): Processor for image transformations and tokenization.
        max_target_length (int): Maximum allowed length for text sequences.
    '''
    
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def preprocess_image(self, image):
        '''
        Preprocess the input image by resizing and padding it to a target size, 
        while maintaining the aspect ratio.
        
        Args:
            image (PIL.Image): Input image to be processed.
        
        Returns:
            PIL.Image: Resized and padded image.
        '''
        target_size = (224, 224)
        original_size = image.size

        aspect_ratio = original_size[0] / original_size[1]
        if aspect_ratio > 1:
            new_width = target_size[0]
            new_height = int(target_size[0] / aspect_ratio)
        else:
            new_height = target_size[1]
            new_width = int(target_size[1] * aspect_ratio)

        resized_img = image.resize((new_width, new_height))

        pad_image = Image.new('RGB', target_size, (255, 255, 255))
        pad_left = (target_size[0] - new_width) // 2
        pad_top = (target_size[1] - new_height) // 2

        pad_image.paste(resized_img, (pad_left, pad_top))
        return pad_image

    def __getitem__(self, idx):
        '''
        Retrieves the item (image-text pair) at the given index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            dict: A dictionary containing preprocessed image tensors and corresponding text labels.
        '''
        try:
            # Get file name and corresponding text
            file_name = self.df.iloc[idx]['file_name']
            text = self.df.iloc[idx]['text']

            # Load and process image
            image_path = os.path.join(self.root_dir, file_name)
            image = Image.open(image_path).convert("RGB")
            image = self.preprocess_image(image)
            pixel_values = self.processor(image, return_tensors="pt").pixel_values

            # Encode text as input IDs (tokenized format)
            labels = self.processor.tokenizer(text, 
                                              padding="max_length", 
                                              max_length=self.max_target_length,
                                              return_tensors="pt").input_ids.squeeze()

            # Ensure PAD tokens are ignored by the loss function
            labels = torch.where(labels == self.processor.tokenizer.pad_token_id, -100, labels)

            return {"pixel_values": pixel_values.squeeze(), "labels": labels}

        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Assuming you have a DataFrame `df` with image file names and text labels
    import pandas as pd
    from transformers import AutoProcessor

    # Example processor
    processor = AutoProcessor.from_pretrained("facebook/trocr-base-handwritten")

    # Example dataset and root directory
    root_dir = "/path/to/dataset/images"
    df = pd.read_csv("/path/to/dataset/annotations.csv")

    # Instantiate the dataset
    dataset = IIITHWDataset(root_dir=root_dir, df=df, processor=processor, max_target_length=128)

    # Access a sample
    sample = dataset[0]
    print(sample)
