# AI Concepts, Terminology and Application Domains (Foundational Concepts of AI)

This repository contains the hands-on notebooks for the second module of the "Introduction to Artificial Intelligence" course, part of the **"IBM Generative AI Engineering"** Specialization on Coursera. These notebooks are the personal study notes and code implementations documented by **Fahad Shah (1FahadShah)**.

This module is a technical deep-dive that moves from the "what" of AI to the "how." It builds the engineering bedrock for the rest of the specialization, covering the core concepts of machine learning, the architectures of deep learning, the mechanics of neural networks, and the high-level systems (like IoT and Edge) that make real-world AI possible.

---

## Notebook Overview 📓

This module is a comprehensive, 8-part series that covers the entire AI engineering stack, from cognitive theory to system-level simulation.

- **`01_Cognitive_Computing_and_AI_Paradigms.ipynb`**
  Establishes the conceptual foundation, comparing static, rule-based logic with an adaptive Cognitive Computing approach. Implements a simple `sklearn` model to show how machines "learn" from data.

- **`02_AI_Terminologies_and_Ecosystem.ipynb`**
  Visually breaks down the complete AI ecosystem (AI > ML > DL > NN). It contrasts a classical ML model (`LogisticRegression`) with a shallow Neural Network (`MLPClassifier`) on the digits dataset.

- **`03_Machine_Learning_Fundamentals_and_Types.ipynb`**
  A hands-on tour of the three core ML paradigms. It implements Supervised Learning (Classification & Regression), Unsupervised Learning (K-Means Clustering), and Reinforcement Learning (a simple Q-learning simulation).

- **`04_Deep_Learning_and_Neural_Networks_Basics.ipynb`**
  The technical deep-dive. This notebook builds a 2-layer Neural Network **from scratch using only NumPy**, implementing activation functions and backpropagation to solve the classic nonlinear XOR problem.

- **`05_Generative_AI_Architectures_and_Types.ipynb`**
  Explores the "creative" models, providing conceptual demos of VAEs (Variational Autoencoders), GANs (Generative Adversarial Networks), autoregressive logic, and the Transformer architecture.

- **`06_Large_Language_Models_and_Foundation_AI.ipynb`**
  Focuses on the engine of modern GenAI. This notebook covers tokenization, vector embeddings, and the self-attention mechanism (visualized with a heatmap), and uses a `transformers` pipeline to generate text.

- **`07_AI_Domains_NLP_Speech_Vision.ipynb`**
  A practical guide to AI's "senses." It implements:
    * NLP: Sentiment analysis with Hugging Face.
    * Speech: Text-to-Speech (TTS) with `pyttsx3`.
    * Vision: Image classification with a pre-trained `ResNet-18`.

- **`08_AI_Cloud_Edge_IoT_Integration_and_Applications.ipynb`**
  A systems-level simulation. This notebook builds an end-to-end IoT -> Edge -> Cloud pipeline in Python, simulating a device stream and using an `IsolationForest` model for real-time anomaly detection.

---

## Key Concepts Covered ✨

This module provides the complete technical toolkit for understanding modern AI, from classical machine learning to the infrastructure that supports it.

#### 1. Foundational Paradigms & Architectures
- Cognitive Computing: Moving beyond automation to systems that Observe, Interpret, Evaluate, and Decide.
- AI vs. ML vs. DL vs. Foundation Models: The complete hierarchy, from the broad field of AI to the specific, large-scale models that power GenAI.
- Machine Learning Paradigms:
    - Supervised: Learning from labeled data (e.g., Classification, Regression).
    - Unsupervised: Finding hidden patterns (e.g., Clustering).
    - Reinforcement: Learning via trial, error, and rewards (e.g., Q-learning).
- Generative Models: The core architectures that create new data: VAEs, GANs, Autoregressive Models, and Transformers.

#### 2. The Core Engine: Deep Learning & NLP
- Neural Networks: The building blocks of deep learning, including Neurons, Weights, Biases, and non-linear Activation Functions (like Sigmoid & ReLU).
- Backpropagation: The "learning" algorithm that adjusts a network's weights to reduce error.
- Natural Language Processing (NLP): The "senses" of AI, including Tokenization, Named Entity Recognition (NER), and Sentiment Analysis.
- Computer Vision (CV): How machines see, using models like CNNs (e.g., ResNet) for classification and detection.

#### 3. Systems & Infrastructure
- IoT (Internet of Things): The network of sensors that collect real-world data.
- Edge Computing: Processing data locally on a device for extremely fast, low-latency decisions (e.g., real-time anomaly detection).
- Cloud Computing: The centralized hub for aggregating data, performing large-scale analytics, and training complex AI models.
- The Synergy: Understanding the trade-offs between instant Edge reflexes and the deep Cloud brain.

---

## How to Use These Notebooks 🛠️

1.  Ensure you have a Python environment (e.g., VS Code, JupyterLab, Google Colab).
2.  Install the required libraries listed in `requirements.txt` (or install them as needed when running the notebooks).
3.  For the best learning experience, open and run the notebooks sequentially from `01` to `08`.
4.  Read the markdown cells for detailed theory and run the code cells to see the concepts in action.

---

## Technologies / Tags
`Python` | `Jupyter` | `Scikit-learn` | `NumPy` | `Pandas` | `PyTorch` | `Transformers` | `Hugging Face` | `spaCy` | `OpenCV` | `pyttsx3` | `Generative AI` | `LLMs` | `Neural Networks` | `Machine Learning` | `Deep Learning` | `NLP` | `Computer Vision` | `IoT` | `Edge Computing`

---
*These notebooks are part of the IBM Generative AI Engineering Specialization, documented and adapted by Fahad Shah (1FahadShah) as part of his personal learning journey.*
