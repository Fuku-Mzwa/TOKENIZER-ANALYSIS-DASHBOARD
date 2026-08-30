
# Extract the positions of the morpheme boundaries from the gold segmentation dataset
def get_gold_boundaries(gold_segmentation):
    # tracks the character position in the original word
    char_count = 0
    
    # Stores the boundary positions. A set is used to ensure each position is stored only once
    boundaries = set()
    
    for char in gold_segmentation:
        if char == '_':
           # An underscore represents a morpheme boundary in the gold standard segmentation
           # The current character count represents the boundary position
           boundaries.add(char_count)
        else:
          # Move to the next character in the original word
           char_count += 1  
    
    return boundaries

# Extract the positions of token boundaries from the tokenizer output
def get_tokenizer_boundaries(tokens):
    # Tracks the number of characters processed so far
    char_count = 0
    
    # Stores the character positions at which the tokenizer places the token boundary
    boundaries = set()
    
    # A boundary exists after every token except the final token, 
    # since there is no boundary after the end of the word
    for token in tokens[:-1]:
        # Add the length of the current token to obtain the position of the boudary following that token
        char_count += len(token)
        boundaries.add(char_count)
        
    return boundaries


# Compare the gold standard morpheme boundaries with the tokenizer's token boundaries
# Then calculate the number of true positives , false positives , false negatives and true negatives
def get_boundary_counts(text , tokens, gold_segmentation):

    # Obtain the true , correct morpheme boundary positions from the gold standard segmentation
    gold_boundaries = get_gold_boundaries(gold_segmentation)
    
    # Obtain the boundary positions produced by the tokenizer
    tokenizer_boundaries = get_tokenizer_boundaries(tokens)
    
    # Generate all possible boundary positions in the original text
    # For a word of length n , there are n-1 possible positions between characters
    possible_positions = set(range(1, len(text)))
    
    # True Positive
    # tokenizer boundary position is the same as the gold standard segamentation boundary position
    tp = len(gold_boundaries & tokenizer_boundaries) 
    
    # False Positive 
    # The tokenizer places a boundary where the gold standard segmentation does not have one
    fp = len(tokenizer_boundaries - gold_boundaries) 
    
    # True Negative
    # Neither the gold standard segmentation nor the tokenizer places a boundary at this position
    tn = len(possible_positions - (gold_boundaries | tokenizer_boundaries))
    
    # False Negative
    # The tokenizer fails to place a boundary at a position where the gold stadard segmentation contains a boundary
    fn = len(gold_boundaries - tokenizer_boundaries) 
    
    return tp,fp, tn , fn


# Calculate Precision.
# Precision measures the proportion of the tokenizer produced boundaries
# that are correct according to the gold standard segmentation
def precision(tp, fp):
    if(tp + fp) == 0:
      return 0.0
    return tp/ (tp + fp)
    
# Calculate Recall.
# Recall measures the proportion of gold standard boundaries that 
# were correctly identified by the tokenizer
def recall(tp , fn):
    if tp + fn == 0:
      return 0.0
    return tp / (tp + fn)

# Calculate the F1 score
# F1 is the harmonic mean of Precision and Recall 
# provides a single measure that balances both metrics
def f1(precision, recall):
    if precision + recall == 0:
       return 0.0
       
    return 2 * (precision * recall) / (precision + recall)
    