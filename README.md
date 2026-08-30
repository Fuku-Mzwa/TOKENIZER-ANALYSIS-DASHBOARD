# Backend Components

This directory contains the backend components responsible for tokenizer loading and metric calculations for the Tokenizer Analysis Dashboard.

1. tokenizer_services.py

#Purpose

Handles the loading and management of Hugging Face tokenizers used by the application

# Main responsibilities
-Loads a tokenizer from the Hugging Face Hub using its model ID
-Provides tokenization functionality for the selected tokenizer
-Uses caching to avoid unnecessarily loading the same tokenizer multiple times
-Acts as the interface between the Streamlit application and the Hugging Face tokenizer library

# Main dependencies
-transformers
-logging

#Used by
The Streamlit frontend will call the tokenizer service when a user selects and loads a tokenizer

2. efficiency_metrics.py

#Purpose
Calculates the efficiency-related metrics used to evaluate tokenizer performance on the default dataset

## Main responsibilities

-Calculates the number of tokens produced by a tokenizer
-Calculates tokenizer fertility
-Calculates compression
-Calculates token premium
-Handles cases such as division by zero where necessary

# Metrics

## Token Count

The total number of tokens produced by the tokenizer for a given text.

# Fertility

Measures the average number of tokens produced per word.

#Compression

Measures how efficiently the tokenizer represents the input text in terms of token count.

# Token Premium

Measures the relative increase in tokenization cost for a language compared with English.

## Used by

The Streamlit application will use these functions to calculate and display tokenizer efficiency results.

3. mbi.py
# Purpose
Calculates morpheme boundary alignment metrics between the tokenizer output and the gold-standard segmentation

# Main responsibilities

- Obtains the gold-standard morpheme boundaries from the dataset
- Determines the boundaries produced by the tokenizer
- Compares predicted tokenizer boundaries with the gold boundaries
- Calculates: Precision Recall F1-score

# Gold-standard boundaries

The default dataset contains morpheme segmentations using `_` to indicate morpheme boundaries

The `_` character itself is not treated as part of the original text. The implementation keeps track of the character positions so that the gold boundaries can be compared with the boundaries produced by the tokenizer

### Metrics

#### Precision

Measures how many of the boundaries identified by the tokenizer are correct

#### Recall

Measures how many of the gold-standard morpheme boundaries were correctly identified by the tokenizer.

#### F1-score

Provides a combined measure of precision and recall.
