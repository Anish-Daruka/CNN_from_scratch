# CNN from Scratch (NumPy) – MNIST Digit Classifier

Implemented a **Convolutional Neural Network (CNN)** from scratch using **only NumPy**. It trains on the **MNIST** dataset of handwritten digits and achieves an accuracy of **92.07% after training on 5 epochs**.


## Model Architecture

Input: 28x28 grayscale image  
→ Conv Layer: 3x3 kernel, 4 filters, stride=1, padding=0 → Output: 26x26x4  
→ ReLU  
→ Max Pooling: 2x2 → Output: 13x13x4  

→ Conv Layer: 3x3 kernel, 8 filters, stride=1, padding=0 → Output: 11x11x8  
→ ReLU  
→ Max Pooling: 2x2 → Output: 5x5x8  

→ Flatten → Dense (200 units) → Output: 10 classes (digits 0–9)


## Hyperparameters

Learning rate: 0.01  
Batch size: 32  
Kernel size: 3  
Stride: 1  
Padding: 0


## Results

Accuracy: 92.07%  
Epochs: 5  
Dataset: MNIST (60,000 train / 10,000 test images)  
