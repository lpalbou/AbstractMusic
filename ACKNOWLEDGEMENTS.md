# Acknowledgements

AbstractMusic builds on the Python scientific and audio generation ecosystem, including PyTorch,
Diffusers, Transformers, Hugging Face Hub, and the model projects exposed through optional
backends.

Model weights, licenses, and usage terms remain governed by their upstream authors. See
[docs/models.md](docs/models.md) for the reviewed model list and license notes.

The internal Stable Audio 3 inference subset in `abstractmusic.vendor.stable_audio3_min` is
adapted from Stability AI's Stable Audio 3 reference implementation, which is MIT licensed.
AbstractMusic carries the upstream license notice with that vendored subset. Stable Audio 3 model
weights remain governed by the Stability AI Community License and the text-encoder terms listed on
the Hugging Face model cards.
