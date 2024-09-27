from evaluate import load

cer_metric = load("cer")


# Function to filter out special tokens and empty strings
def filter_special_tokens(decoded_strs,processor ):
    filtered_strs = []
    for s in decoded_strs:
        # Remove special tokens <s>, </s>, and <pad>
        filtered = s.replace(processor.tokenizer.bos_token, "") \
                    .replace(processor.tokenizer.eos_token, "") \
                    .replace(processor.tokenizer.pad_token, "")
        filtered = filtered.strip()  # Remove extra spaces
        if filtered:  # Only include non-empty strings
            filtered_strs.append(filtered)
    return filtered_strs

# Function to pad the shorter list to match the length of the longer one
def pad_to_equal_length(label_str, pred_str,processor):
    max_len = max(len(label_str), len(pred_str))
    # Pad the shorter list with the pad_token to match the length of the longer list
    label_str.extend([processor.tokenizer.pad_token] * (max_len - len(label_str)))
    pred_str.extend([processor.tokenizer.pad_token] * (max_len - len(pred_str)))
    return label_str, pred_str


def compute_metrics(pred,processor):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    
    # Strip unnecessary whitespace that may be introduced after decoding
    pred_str = filter_special_tokens(pred_str,processor)
    label_str = filter_special_tokens(label_str,processor)
    

    # padd for equal length
    label_str,pred_str = pad_to_equal_length(pred_str,label_str)
    
    # print('label_string',label_str)
    # print('prediction string',pred_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}