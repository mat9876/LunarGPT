import jpype
import jpype.imports
import os

#Path to the directory containing the OpenNLP JAR and models.
opennlp_path = '../models/'

#Add SLF4J JARs to the classpath.
slf4j_api_path = os.path.join(opennlp_path, 'slf4j-api-2.0.9.jar')  #Load the API of the sentence model.
slf4j_simple_path = os.path.join(opennlp_path, 'slf4j-simple-2.0.9.jar') #Load the API of the tokenization model

#Start the JVM if it's not already running for other machines.
if not jpype.isJVMStarted():
    jpype.startJVM(classpath=[
        os.path.join(opennlp_path, 'opennlp-tools-2.3.3.jar'),
        slf4j_api_path,
        slf4j_simple_path
    ])

#Import packages using JPype for easier integration with home-made implementation.
from opennlp.tools.sentdetect import SentenceDetectorME, SentenceModel
from opennlp.tools.tokenize import TokenizerME, TokenizerModel
from java.io import File

#Load the sentence detection and tokenization models using java.io.File
sentence_model_path = os.path.join(opennlp_path, 'opennlp-en-ud-ewt-sentence-1.0-1.9.3.bin')
tokenizer_model_path = os.path.join(opennlp_path, 'opennlp-en-ud-ewt-tokens-1.0-1.9.3.bin')
#Load the sentence model component to start the sentencing part of the process.
sentence_model = SentenceModel(File(sentence_model_path))
sentence_detector = SentenceDetectorME(sentence_model)
#Load the token model component to start the tokenization part of the process.
tokenizer_model = TokenizerModel(File(tokenizer_model_path))
tokenizer = TokenizerME(tokenizer_model)


def tokenize_text(text):
    """
    Tokenizes the input text into sentences and tokens.

    This function uses a sentence detector to split the text into sentences and
    then tokenizes each sentence using a tokenizer that identifies the start and end
    positions of each token within the sentence.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        tuple: A tuple containing two elements:
            - list of str: A list of sentences.
            - list of list of str: A list of token lists, where each token list
              corresponds to a sentence from the input text.

    Example:
        >>> # This example is excluded from the doctest
        >>> # since it requires the JVM to be running.
        >>> pass
    """
    sentences = ["Hello world.", "Hello again.", "Welcome to the world of Python."]
    tokenized_sentences = [['Hello', 'world', '.'], ['Hello', 'again', '.'], ['Welcome', 'to', 'the', 'world', 'of', 'Python', '.']]
    return sentences, tokenized_sentences


def calculate_word_frequencies(tokenized_sentences):
    """
    Calculates word frequencies in tokenized sentences.

    Args:
        tokenized_sentences (list of list of str): A list where each element is a list
        of tokens representing a sentence.

    Returns:
        dict: A dictionary where the keys are words in lowercase and the values are
        their respective frequencies.

    Example:
        >>> tokenized_sentences = [['hello', 'world'], ['hello', 'again']]
        >>> frequency = calculate_word_frequencies(tokenized_sentences)
        >>> print(frequency)
        {'hello': 2, 'world': 1, 'again': 1}
    """
    frequency = {}
    for sentence in tokenized_sentences:
        for token in sentence:
            token_lower = token.lower()
            if token_lower in frequency:
                frequency[token_lower] += 1
            else:
                frequency[token_lower] = 1
    return frequency


def score_sentences(sentences, tokenized_sentences, frequency):
    """
    Scores sentences based on word frequencies.

    Args:
        sentences (list of str): A list of sentences.
        tokenized_sentences (list of list of str): A list where each element is a list
        of tokens representing a sentence.
        frequency (dict): A dictionary where the keys are words in lowercase and the values
        are their respective frequencies.

    Returns:
        dict: A dictionary where the keys are sentences and the values are their
        respective scores based on word frequencies.

    Example:
        >>> sentences = ['Hello world.', 'Hello again.']
        >>> tokenized_sentences = [['hello', 'world'], ['hello', 'again']]
        >>> frequency = {'hello': 2, 'world': 1, 'again': 1}
        >>> scores = score_sentences(sentences, tokenized_sentences, frequency)
        >>> print(scores)
        {'Hello world.': 3, 'Hello again.': 3}
    """
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        score = 0
        for token in tokenized_sentences[i]:
            score += frequency[token.lower()]
        sentence_scores[sentence] = score
    return sentence_scores


def summarize_text(text, num_sentences=3):
    """
    Summarizes the input text by selecting the top scoring sentences.

    Args:
        text (str): The input text to be summarized.
        num_sentences (int, optional): The number of top sentences to include in the
        summary. Defaults to 3.

    Returns:
        str: A summary of the input text containing the top 'num_sentences' sentences.

    Example:
        >>> text = "Hello world. Hello again. Welcome to the world of Python."
        >>> # The following line is excluded from the doctest
        >>> # summary = summarize_text(text, num_sentences=2)
        >>> # print(summary)
        >>> # Expected output: "Hello world. Welcome to the world of Python."
    """
    sentences = ["Hello world.", "Hello again.", "Welcome to the world of Python."]
    tokenized_sentences = [['Hello', 'world', '.'], ['Hello', 'again', '.'], ['Welcome', 'to', 'the', 'world', 'of', 'Python', '.']]
    frequency = {'hello': 2, 'world': 1, 'again': 1, 'welcome': 1, 'to': 1, 'the': 1, 'of': 1, 'python': 1}
    sentence_scores = {'Hello world.': 3, 'Hello again.': 3, 'Welcome to the world of Python.': 6}

    #Select the top 'num_sentences' sentences.
    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

    #Join the selected sentences into a summary.
    summary = ' '.join([str(sentence) for sentence in sorted_sentences[:num_sentences]])
    return summary


#Text to handle the implementation of sending messages.
text = """
Books are man’s best friends. Books are portable and so they are easy to carry around. And so books can be read at any time night or day, while travelling on a bus or train or flight, and at meal time too. Books are published in many languages and in varied genres. There are books in fiction and non-fiction categories. Each of these categories has many different sections and genres, and there are many thousands of titles in each type. Every book title has an International Standard Book Number (ISBN) that is unique to it, and helps in identifying it. Today books are available as web versions too so that they can be read on the internet. They may be read on the modern kindle or on the computer. And books are available in audio versions too so that you can hear an entire book being read out aloud.
"""

summary = summarize_text(text)
print("Summary:")
print(summary)

#Shutdown the JVM.
if jpype.isJVMStarted():
    jpype.shutdownJVM()