# Functions for calculating efficiency metrics from tokenized text

#Determining number of tokens produced
def token_count(tokens):
    token_count = len(tokens)
    return token_count
    
# Calculating the average number of tokens produced per word
def fertility(text , tokens):
    
    # Number of tokens 
    token_count  = len(tokens)
    
    #Count number of words in a given text
    word_count = len(text.split())
    
    if word_count == 0:
       return 0.0
       
    #Return fertilility
    return token_count / word_count
   
# Ratio of bytes per token
def compression(text , tokens):
    # Number of tokens 
    token_count  = len(tokens)
    
    #Get the number of bytes in the given text
    bytes_count = len(text.encode('UTF-8')) # UTF-8 supports the Unicode characters used across all the supported languages
    
    if token_count == 0:
      return 0.0
    
    return bytes_count/token_count
    
# English is used as the baseline for calculating token premium
def token_premium(tokens, english_tokens):
    # Number of tokens for the target language
    token_count = len(tokens)

    # Number of tokens for English
    token_English = len(english_tokens)

    if token_English == 0:
        return 0.0

    return token_count / token_English

    
    
    
    