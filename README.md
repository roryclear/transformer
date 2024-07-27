# Transformer
## OpenCL and Metal GPT-2 inference
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
| huggingface/transformers  |53 t/s         |17 t/s         |8 t/s      |  
| **roryclear/transformer** |**31 t/s**     |**15 t/s**     |**8 t/s**  |

|Intel Integrated Graphics (2020 XPS13)         | GPT2          |GPT2-Medium    |GPT2-Large |
| -----------                                   | -----------   |------         |----       |
| tinygrad                                      |16 t/s         |5.8 t/s        |2.1 t/s    |
| huggingface/transformers                      |34 t/s         |15 t/s         |7.7 t/s    |  
|**roryclear/transformer**                      |**6.3 t/s**    |**2.0 t/s**    |**1.0 t/s**|

*producing 100 tokens from 13 token prompt
