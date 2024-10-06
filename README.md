# **Handwritten Document Conversion**

## **Introduction**

This project focuses on converting handwritten documents, particularly in Nepali script, into digital text using a state-of-the-art (SOTA) model. The system automates the extraction of handwritten text from single-page documents, significantly improving efficiency and accuracy in document processing.

## **Goals**

- Develop a high-accuracy OCR model for handwritten Nepali text recognition.
- Automate the conversion of scanned handwritten documents into structured digital text.
- Provide a system that can handle various layouts and document types.

## **Contributors**

- **[Your Name]**: Lead Developer
- **[Contributor's Name]**: Data Scientist
- **[Contributor's Name]**: UI/UX Designer

## **Project Architecture**

The architecture leverages modern deep learning techniques, incorporating state-of-the-art models like TrOCR for text recognition and LayoutLMv3 for document layout analysis. The system processes scanned images, extracts text, and outputs structured data ready for use.

---

# **Status**

- Current phase: **Model Training and Optimization**

## **Known Issues**

- Minor inaccuracies in detecting certain handwritten styles.
- Overfitting on specific types of documents with complex layouts.

## **High-Level Next Steps**

- Fine-tune the model to handle additional handwritten styles.
- Expand the system to support multi-page document conversion.
- Test the system on a wider variety of documents from different sectors.

---

# **Usage**

## **Installation**

To begin using this project, you can utilize the included `Makefile`.

### **Creating Virtual Environment**

This project requires `python-3.8`. To ensure compatibility, we recommend creating a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
```

### **Pre-commit**

Install and activate `pre-commit` to automatically format and lint your code.

```bash
make use-pre-commit
```

It will run every time you commit changes via Git.

### **pip-tools**

Manage dependencies using `pip-tools` by running:

```bash
make use-pip-tools
```

To install dependencies or update new ones, modify the `requirements.in` file, then:

```bash
make deps-install
# Or simply:
make
```

To sync and clean unused dependencies:

```bash
make deps-sync
```

---

## **Usage Instructions**

Follow the below instructions to run the system and test it on your documents:

1. Upload a scanned handwritten document.
2. Run the system to extract the handwritten text.
3. View the results in digital format.

---

# **Data Source**

- The system uses a custom dataset with handwritten Nepali text, both printed and annotated.
- Source data includes documents from various sectors such as education and government.

## **Code Structure**

- **/src**: Contains the core processing scripts.
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
