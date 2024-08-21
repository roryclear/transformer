# Transformer
## Running:
git clone --depth 1 https://github.com/roryclear/transformer.git

pip install -r requirements.txt

pip install pycuda (CUDA only)

pip install pyobjc-framework-Metal (Apple Silicon only)

python demo.py --p="your prompt" 

## CUDA, OpenCL and Metal GPT-2 inference
- [ ] Fastest Inference

- [X] Support for all GPT-2 Models

- [X] Multiple Compute Languages

- [X] Remove all Numpy usage (For calculations)

- [ ] Support any transformer

## Performance:

### CUDA
|T4 x2 (Kaggle Cloud)         | GPT2          |GPT2-Medium    |GPT2-Large |
| -----------                 | -----------   |------         |----       |
| tinygrad                    |74 t/s         |39 t/s         |24 t/s     |
| huggingface/transformers    |30 t/s         |12 t/s         |6.1 t/s    |  
|**roryclear/transformer**    |**71 t/s**     |**28 t/s**     |**13 t/s** |

|P100 (Kaggle Cloud)          | GPT2          |GPT2-Medium    |GPT2-Large |
| -----------                 | -----------   |------         |----       |
| tinygrad                    |57 t/s         |31 t/s         |20 t/s     |
| huggingface/transformers    |31 t/s         |12 t/s         |6.0 t/s    |  
|**roryclear/transformer**    |**59 t/s**     |**21 t/s**     |**10 t/s** |

### Metal
|Apple M2                   | GPT2          |GPT2-Medium    |GPT2-Large |
| -----------               | -----------   |------         |----       |
| tinygrad                  |30 t/s         |22 t/s         |15 t/s     |
| huggingface/transformers  |53 t/s         |17 t/s         |8 t/s      |  
| **roryclear/transformer** |**33 t/s**     |**16 t/s**     |**9 t/s**  |

### OpenCL
|Intel Integrated Graphics (2020 XPS13)         | GPT2          |GPT2-Medium    |GPT2-Large |
| -----------                                   | -----------   |------         |----       |
| tinygrad                                      |16 t/s         |5.8 t/s        |2.1 t/s    |
| huggingface/transformers                      |34 t/s         |15 t/s         |7.7 t/s    |  
|**roryclear/transformer**                      |**10 t/s**     |**2.5 t/s**    |**2.0 t/s**|

*generating 100 tokens from a 13 token prompt, I don't own any Nvidia hardware to measure CUDA speeds properly.
