# FNet

PyTorch implementation of [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824v1).

<p align="center">
  <img src="https://miro.medium.com/max/1551/0*LE7Bqa1C-JIAWP7Z.png">
</p>

## Quickstart

Clone this repository.

```
git clone https://github.com/jaketae/fnet.git
```

Navigate to the cloned directory. You can start using the model via

```python
>>> from fnet import FNet
>>> model = FNet()
```

By default, the model comes with the following parameters:

```python
FNet(
    d_model=256,
    expansion_factor=2,
    dropout=0.5,
    num_layers=6,
)
```

## Summary

While transformers have proven to be successful in various domains, its `O(n^2)` computation complexity has been considered a structural weakness. Many attempts have been made to optimize the model architecture. The authors of the paper present FNet, a model that replaces self-attention with standard unparametrized Fourier transforms. Not only is FNet faster and computationaly more efficient than the classic transformer, but it also retains 92% of BERT's accuracy on the GLUE benchmark. Given a smaller number of parameters, FNet outperformed transformers.

## Resources

-   [Original Paper](https://arxiv.org/abs/2105.03824v1)
-   [Rishikesh's implementation](https://github.com/rishikksh20/FNet-pytorch)
-   Image from [SyncedReview's Medium article](https://medium.com/syncedreview/google-replaces-bert-self-attention-with-fourier-transform-92-accuracy-7-times-faster-on-gpus-7a78e3e4ac0e)
