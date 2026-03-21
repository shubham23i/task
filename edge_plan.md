# Edge Deployment Plan

## Goal

The goal of this system is to operate efficiently in real-world environments, including mobile and offline settings. This requires careful consideration of model size, latency, and computational constraints.

## Model Selection

The current system uses TF-IDF combined with classical machine learning models. This approach was chosen because it provides a strong balance between performance and efficiency. Unlike deep learning models, it does not require heavy computational resources and can run on standard devices.

## Optimization Strategy

To make the system suitable for edge deployment, several optimizations can be applied. The number of TF-IDF features can be reduced to lower memory usage, and simpler models such as Logistic Regression can be preferred for faster inference. Model files can also be compressed or converted to formats like ONNX for better portability.

## Performance

The system achieves low latency, typically under 100 milliseconds per prediction, making it suitable for real-time applications. Memory usage is also kept within reasonable limits, allowing deployment on devices with constrained resources.

## Deployment Flow

The on-device pipeline consists of taking user input, transforming it using a pre-trained TF-IDF vectorizer, generating predictions for emotional state and intensity, and finally passing the results through the decision engine to produce actionable guidance.

## Trade-offs

There is a trade-off between model complexity and efficiency. While deep learning models could improve accuracy, they would significantly increase latency and resource consumption. The current approach prioritizes speed and reliability over maximum accuracy.

## Conclusion

The chosen architecture ensures that the system remains lightweight, fast, and practical for deployment in real-world scenarios, particularly where internet connectivity or computational power is limited.
