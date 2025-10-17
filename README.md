# Service Desk Classifier

This repository contains a Service Desk ticket classification system with two main components:

```
sd-classifier/
  app/                 # FastAPI web app for inference, serving, and LLM-based triage
  classifier_train/    # Model training code (data preparation, training loop, evaluation)
```

* **`app/`**: A production-ready FastAPI application. It exposes REST and HTML endpoints for classifying Service Desk tickets, integrating with an LLM, and retrieving assignment group information. This is the part deployed via Docker/Kubernetes.
* **`classifier_train/`**: Experimental and training code. Includes scripts for preparing data, fine-tuning models, and evaluating performance. Not included in deployment images.


Note: The trained model weights are not included in this repository due to size constraints and ownership by UC San Diego ITS. The code here demonstrates the architecture, data flow, and deployment setup used in production.

See the [`app/README.md`](./app/README.md) for detailed usage, API docs, and deployment instructions.

---
