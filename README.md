# Neural Bandits
In this project we will develop a Multi-Armed Bandits library in python that specializes on contextual neural bandit methods.

[Link to Agreement](https://docs.google.com/document/d/1qs0hDGVd5MHe6PK5uL_GVNjiIePBJscbNkjGotF9-Uk/edit?tab=t.0])

## Architecture Overview
- Network Architecture (pytorch module)
- Exploration Strategy (different Bandits)
- Vector Feedback (also multinomial)
- Training / Optimization
For more details see ARCHITECTURE.md.

## MVP
### Baseline Algorithms
- [ ] LinUCB
- [ ] Linear Thompson Sampling

### Neural Algorithms
- [ ] NeuralUCB
- [ ] NeuralLinear
- [ ] NeuralTS
- [ ] Bootstrap
- [ ] Combinatorial Bandits

### Development Goals:
- The library should be extendable for implementing further methods and models.
- The trained model should be able to work stand-alone for inference => decoupling of Model and Trainer.
- The library is built on top of pytorch => each model will be a `torch.nn.Module`.

### Evaluation Datasets
- [x] MNIST
- [x] Statlog
- [x] Covertype
- [x] Wheel Bandit
- [ ] RAG

### Further Optional Directions
- [ ] Norm-POEM style on logged bandit data
- [ ] LoRA style updates
