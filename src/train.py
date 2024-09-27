import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoProcessor
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel
from transformers import default_data_collator
from functools import partial

import pandas as pd


from src.trocr import HandwrittenTextModel, HandwrittenTextProcessor
from src.dataset import IIITHWDataset
from src.scripts import compute_metrics

# set up a device
# define the file paths
train_paths = "data/train_extracted_files.csv"
test_paths= "data/test_extracted_files.csv"
val_paths = "data/val_extracted_files.csv"
root_dir = "data/IIIT-HW-Hindi_v1/"




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






def main():
    #load the dataset in Dataframe format
    train_df = pd.read_csv(train_paths)[0:10]
    test_df = pd.read_csv(test_paths)[0:10]
    val_df = pd.read_csv(val_paths)[0:10]

    print(f"Train,Test and Val shapes: {train_df.shape, test_df.shape, val_df.shape}")

    #Intialize the processor and model
    #initialize processor
    processor= HandwrittenTextProcessor(encode_model='google/vit-base-patch16-224-in21k', decode_model='amitness/nepbert').get_processor()






    train_dataset = IIITHWDataset(root_dir = root_dir,
                            df= train_df,
                            processor = processor,
                            max_target_length=16
                            )
    eval_dataset = IIITHWDataset(root_dir = root_dir,
                                df= val_df,
                                processor = processor,
                                max_target_length=16
                                )
    
    model = HandwrittenTextModel(encode_model='google/vit-base-patch16-224-in21k', decode_model='amitness/nepbert')
    model.configure_model(processor, config={'max_length':16})
    model = model.get_model()

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(eval_dataset))

    
    # Define training arguments

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",  # Evaluate every few steps
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        output_dir='model/checkpoints',  # Output directory to save models
        save_total_limit=2,  # Limit the number of saved checkpoints
        logging_steps=2,  # Log every 2 steps
        save_steps=1000,  # Save model every 100 steps
        eval_steps=1000,  # Evaluate every 100 steps
        save_strategy="steps",  # Save based on steps
        load_best_model_at_end=True,  # Load the best model at the end
        metric_for_best_model="cer",  # Use CER to evaluate the best model
        greater_is_better=False,  # Lower CER is better, hence set to False
        num_train_epochs = 5
    )

    compute_metrics_with_processor = partial(compute_metrics, processor=processor)

    # Instantiate the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.tokenizer,  # Ensure the correct tokenizer is used
        args=training_args,
        compute_metrics=compute_metrics_with_processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
   
if __name__ == '__main__':
    main()