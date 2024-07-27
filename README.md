# Transformer
## Fast OpenCL and Metal GPT-2 inference
- [ ] Fastest Inference

- [X] Support for all GPT-2 Models

- [X] Multiple Compute Languages

- [X] Remove all Numpy usage (For calculations)

- [ ] Support for any transformer

- [ ] Refactor to library

### Performance:
|Apple M2 (Metal)           | GPT2          |GPT2-Medium    |GPT2-Large |
| -----------               | -----------   |------         |----       |
| tinygrad                  |32 t/s         |23 t/s         |16 t/s     |
| huggingface/transformers  |60 t/s         |19 t/s         |9 t/s      |  
| roryclear/transformer     |31 t/s         |15 t/s         |8 t/s      |

*producing 100 tokens from 13 token prompt
