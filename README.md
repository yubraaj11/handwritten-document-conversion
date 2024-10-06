# **Handwritten Document Conversion**

## **Introduction**

This project focuses on converting handwritten documents, particularly in Nepali script, into digital text using a state-of-the-art (SOTA) model. The system automates the extraction of handwritten text from single-page documents, significantly improving efficiency and accuracy in document processing.

## **Goals**

- Develop a high-accuracy OCR model for handwritten Nepali text recognition.
- Automate the conversion of scanned handwritten documents into structured digital text.
- Provide a system that can handle various layouts and document types.

## **Contributors**

- **[Aayush Puri]**
- **[Anil Paudel]**
- **[Yubraj Sigdel]**

## **Project Architecture**

The architecture leverages modern deep learning techniques, incorporating state-of-the-art models like TrOCR with ViT for encoder and RoBERTa for decoder. The system processes scanned images, extracts text, and outputs digitized data ready for use.

---

# **Status**

- Current phase: **Model Deployment**

## **Known Issues**

- Minor inaccuracies in detecting certain handwritten styles.
- Overfitting on specific types of documents with complex layouts.

## **High-Level Next Steps**

- Fine-tune the model to handle additional handwritten styles.
- Expand the system to support multi-page document conversion.
- Test the system on a wider variety of documents from different sectors.

---

# **Usage**

## **Creating Virtual Environment**

This project requires `python-3.8`. To ensure compatibility, we recommend creating a virtual environment.

```bash
conda create -n handwritten python==3.10
conda activate handwritten
```

## **Pulling Repository**

### For Linux

```commandline
git clone git@github.com:fuseai-fellowship/hand-written-document-conversion.git
```

### For Windows

```commandline
git clone https://github.com/fuseai-fellowship/hand-written-document-conversion.git
```

## **Install required requirements**

```commandline
pip install -r requirements.txt
```

## Run via gradio app

```commandline
python app.py
```

---

The sample UI is as shown:
(Delete this and paste the ui screenshot via update readme via github)

---

## **Usage Instructions**

Follow the below instructions to run the system and test it on your documents:

1. Upload a scanned handwritten document.
2. Run the system to extract the handwritten text.
3. View the results in digital format displayed beside the image input.

---

# **Data Source**

- The system uses a custom dataset with handwritten Nepali text, both printed and annotated.
- Source data includes documents from various sectors such as education and government.

## **Code Structure**

- **/src**: Contains the core processing scripts.
- **/motebook**: Contains the notebook used while finetuning TrOCR model.
- **/models**: Includes the pre-trained models used in text recognition.
- **/data**: Houses training and test datasets.

## **Artifacts Location**

- Output files and extracted texts are stored in the `/output` directory.
- Checkpoints and trained models are saved under `/models/checkpoints`.

---

# **Results**

## **Metrics Used**

- **Character Error Rate (CER)**: Measures accuracy in recognizing handwritten characters.
- **Word Error Rate (WER)**: Evaluates word-level accuracy.

## **Evaluation Results**

- The system achieved a **CER of X%** and a **WER of X%** on the test set.
- These results demonstrate the model’s ability to generalize across different handwriting styles.
