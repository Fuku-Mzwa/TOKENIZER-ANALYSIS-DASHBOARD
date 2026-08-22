

#A function to return number of tokens give the text and user's choice of the tokenizer
def token_count(text , tokenizer):
   #Apply the selected tokenizer on the given text and get the number of tokens produced
    token_count = len(tokenizer.tokenize(text))
    return token_count
    
 #Counting number of tokens produced by a specified tokenizer per word
def fertility(text , tokenizer):
    #Apply the selected tokenizer on the given text
    token_count  = len(tokenizer.tokenize(text))
    
    #Counting number of words in a guven text
    word_count = len(text.split())
    
    #Return fertilility
    return token_count / word_count
    
def 
    
    
    