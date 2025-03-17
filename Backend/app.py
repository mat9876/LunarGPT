import jpype
import jpype.imports
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to the directory containing the OpenNLP JAR and models
opennlp_path = '../models/'

# Add SLF4J JARs to the classpath
slf4j_api_path = os.path.join(opennlp_path, 'slf4j-api-2.0.9.jar')
slf4j_simple_path = os.path.join(opennlp_path, 'slf4j-simple-2.0.9.jar')

# Start the JVM if it's not already running
if not jpype.isJVMStarted():
    jpype.startJVM(classpath=[
        os.path.join(opennlp_path, 'opennlp-tools-2.3.3.jar'),
        slf4j_api_path,
        slf4j_simple_path
    ])

# Import Java packages using JPype
from opennlp.tools.sentdetect import SentenceDetectorME, SentenceModel
from opennlp.tools.tokenize import TokenizerME, TokenizerModel
from java.io import File

# Load the sentence detection and tokenization models using java.io.File
sentence_model_path = os.path.join(opennlp_path, 'opennlp-en-ud-ewt-sentence-1.0-1.9.3.bin')
tokenizer_model_path = os.path.join(opennlp_path, 'opennlp-en-ud-ewt-tokens-1.0-1.9.3.bin')

sentence_model = SentenceModel(File(sentence_model_path))
sentence_detector = SentenceDetectorME(sentence_model)

tokenizer_model = TokenizerModel(File(tokenizer_model_path))
tokenizer = TokenizerME(tokenizer_model)

# History to store previous texts and summaries
history = []

def tokenize_text(text):
    try:
        sentences = sentence_detector.sentDetect(text)
        tokenized_sentences = []
        for sentence in sentences:
            token_spans = tokenizer.tokenizePos(sentence)
            tokens = [str(sentence[span.getStart():span.getEnd()]) for span in token_spans]
            tokenized_sentences.append(tokens)
        return sentences, tokenized_sentences
    except Exception as e:
        return None, str(e)


def calculate_word_frequencies(tokenized_sentences):
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
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        score = 0
        for token in tokenized_sentences[i]:
            score += frequency.get(token.lower(), 0)
        sentence_scores[str(sentence)] = score  # Ensure sentence is a Python string
    return sentence_scores


def summarize_text(text, num_sentences=3):
    try:
        sentences = sentence_detector.sentDetect(text)
        tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
        # Convert Java strings to Python strings
        tokenized_sentences = [[str(token) for token in sentence] for sentence in tokenized_sentences]
        frequency = calculate_word_frequencies(tokenized_sentences)
        sentence_scores = score_sentences(sentences, tokenized_sentences, frequency)

        sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
        summary = ' '.join([str(sentence) for sentence in sorted_sentences[:num_sentences]])
        return summary, None
    except Exception as e:
        return None, str(e)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    summary, error = summarize_text(text)

    if error:
        # Print the error in the backend console
        print("Error:", error)
        return jsonify({'error': error, 'summary': '', 'history': history})
    else:
        # Print the summary in the backend console
        print("Summary:", summary)
        # Add to history
        history.append({'text': text, 'summary': summary})
        return jsonify({'error': '', 'summary': summary, 'history': history})


if __name__ == '__main__':
    app.run(debug=True)
