# NER Examples

This directory contains examples demonstrating how to use the enhanced NER class with both synchronous and asynchronous support.

## File `ner_demo.ipynb`
Interactive Jupyter notebook with step-by-step examples and explanations.

## NER Class API

### Synchronous Methods
```python
from explain.enhanced_chat.ner import NER

ner = NER()

# Method 1: Direct call
result = ner("BRCA1 is a tumor suppressor gene")

# Method 2: Explicit sync method
result = ner.verify("TP53 encodes p53 protein")
```

### Asynchronous Methods
```python
import asyncio
from explain.enhanced_chat.ner import NER

async def main():
    ner = NER()
    
    # Method 1: Async verify
    result = await ner.averify("EGFR mutations in lung cancer")
    
    # Method 2: Async direct call
    result = await ner.__acall__("KRAS drives many cancers")

asyncio.run(main())
```

### Concurrent Processing
```python
import asyncio
from explain.enhanced_chat.ner import NER

async def process_batch():
    ner = NER()
    texts = [
        "BRCA1 mutations increase breast cancer risk",
        "TP53 is the guardian of the genome",
        "EGFR inhibitors are used in cancer therapy"
    ]
    
    # Process all texts concurrently
    tasks = [ner.averify(text) for text in texts]
    results = await asyncio.gather(*tasks)
    
    return results

results = asyncio.run(process_batch())
```

## Authentication

When you first run any of these examples, you may be prompted to authenticate via your browser. This is normal and required to access the NER service.

## Performance Tips

1. **Use async for batch processing** - Much faster when processing multiple texts
2. **Use concurrent processing** - Process multiple texts simultaneously with `asyncio.gather()`
3. **Use sync for simple scripts** - Easier for quick tests and interactive use
