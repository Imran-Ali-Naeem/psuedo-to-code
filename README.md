# GPT-2 LoRA Fine-Tuning for Pseudocode-to-Code Generation

A fine-tuned GPT-2 model that generates source code from natural language pseudocode using Parameter-Efficient Fine-Tuning (LoRA).

This project uses the SPOC (Synthetic Pseudocode to Code) dataset and applies LoRA adapters on GPT-2 to reduce training cost while maintaining strong code generation capabilities.

---

## 🚀 Overview

Converting pseudocode into executable source code is an important task in program synthesis and AI-assisted software development.

This project fine-tunes GPT-2 on paired pseudocode-code examples from the SPOC dataset, enabling the model to learn mappings between algorithmic descriptions and their corresponding implementations.

---

## ✨ Features

* GPT-2 based code generation
* LoRA (Low-Rank Adaptation) fine-tuning
* Program-level SPOC dataset preprocessing
* Custom pseudocode-to-code dataset creation
* Memory-efficient training
* Automated tokenization pipeline
* Model export and deployment ready

---

## 🏗️ Model Architecture

### Base Model

* GPT-2

### Fine-Tuning Method

* LoRA (Low-Rank Adaptation)

### LoRA Configuration

| Parameter      | Value          |
| -------------- | -------------- |
| Rank (r)       | 8              |
| Alpha          | 32             |
| Dropout        | 0.1            |
| Bias           | None           |
| Target Modules | c_attn, c_proj |

---

## 📂 Dataset

### SPOC Dataset

The project uses the SPOC dataset containing:

* Natural language pseudocode
* Corresponding source code
* Competitive programming solutions
* Program-level code samples

### Data Processing Pipeline

1. Download SPOC dataset
2. Group program lines by problem ID and submission ID
3. Merge pseudocode lines into complete programs
4. Merge code lines into complete solutions
5. Create training pairs

Example format:

```text
### Pseudocode:
Read integer n
Loop from 1 to n
Print the value

### Code:
for(int i=1;i<=n;i++){
    cout << i << endl;
}
```

---

## 📁 Project Structure

```text
project/
├── notebook.ipynb
├── README.md
├── requirements.txt
```

---

## ⚙️ Training Configuration

| Parameter           | Value |
| ------------------- | ----- |
| Model               | GPT-2 |
| Fine-Tuning         | LoRA  |
| Epochs              | 2     |
| Max Sequence Length | 1024  |
| Optimizer           | AdamW |
| Batch Size          | 4     |
| Validation Split    | 10%   |

---

## 🔧 Technologies Used

* Python
* PyTorch
* Transformers
* PEFT (LoRA)
* Hugging Face
* NumPy
* Pandas
* Matplotlib

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/gpt2-pseudocode-to-code.git
cd gpt2-pseudocode-to-code
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Training

Run the notebook:

```bash
jupyter notebook
```

Open:

```text
notebook.ipynb
```

and execute all cells.

---

## 🎯 Inference

Input:

```text
Read an integer n
Print numbers from 1 to n
```

Generated Output:

```cpp
for(int i = 1; i <= n; i++){
    cout << i << endl;
}
```

---

## 📊 Evaluation

The project evaluates model performance using:

* Validation Loss
* Token Prediction Accuracy
* BLEU Score (SacreBLEU)

---

## 💾 Model Saving

The trained model and tokenizer are exported using:

```python
model.save_pretrained("./spoc_gpt2_finetuned")
tokenizer.save_pretrained("./spoc_gpt2_finetuned")
```

---

## 🔮 Future Improvements

* Upgrade to CodeLlama or DeepSeek-Coder
* Train on larger code generation datasets
* Add beam search decoding
* Deploy as a Hugging Face Space
* Integrate with VS Code extension

---

## 📚 References

### SPOC Dataset

Synthetic Pseudocode to Code Dataset

### GPT-2

Radford et al. (2019)

### LoRA

Hu et al. (2021)

https://arxiv.org/abs/2106.09685

---

## 👨‍💻 Author

**Imran Ali**

Computer Science Student | Machine Learning & Generative AI Enthusiast

If you found this project useful, consider giving it a ⭐ on GitHub.
