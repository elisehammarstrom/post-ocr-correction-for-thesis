# Post OCR Correction

This code was used to train a new post-OCR correction model for Swedish that
can be downloaded here <https://huggingface.co/KBLab/swedish-ocr-correction>

The model and implementations are based on [Post-OCR Correction of Digitized
Swedish Newspapers with ByT5](https://aclanthology.org/2024.latechclfl-1.23/)
whose original model can be downloaded
[here](https://huggingface.co/viklofg/swedish-ocr-correction).

## Data

The data used to train the model is described in  [A Two-OCR Engine Method for
Digitized Swedish
Newspapers](https://ecp.ep.liu.se/index.php/clarin/article/view/8) and is
partially available via Språkbanken Text.
The more recent annotated newspapers are not publicly available due to
copyright restrictions.

- [Swedish newspapers 1818-1870](https://spraakbanken.gu.se/en/resources/svenska-tidningar-1818-1870)
- [Swedish newspapers 1871-1906](https://spraakbanken.gu.se/resurser/svenska-tidningar-1871-1906)

## Results

| Model | CER | WER |
| - | - | - |
| Original OCR | 3.01 | 13.23 |
| viklofg |  1.92 | 7.41 |
| KBLab | 1.57 | 6.23 |
