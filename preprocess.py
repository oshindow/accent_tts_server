import jieba
import re
import argparse
import random

def clean_and_split(text):
    # Remove all punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Clean the input text (remove other unwanted characters, spaces, etc.)
    clean_text = text.strip()

    # Split the text into words using jieba
    words = jieba.lcut(clean_text)

    # Insert "sil" between words
    processed_text = ' '.join(word + (' SIL' if random.random() < 0.2 else '') for word in words[:-1]) + ' ' + words[-1]

    return processed_text

def load_lexicon(lexicon_path):
    lexicon = {}
    with open(lexicon_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            word = parts[0]
            pinyin = ' '.join(parts[2:])  # Ignore the first two columns
            if word not in lexicon:
                lexicon[word] = pinyin
    return lexicon

def text_to_pinyin(text, lexicon):
    # words = jieba.lcut(text)
    pinyin_list = []

    for word in text.split(' '):
        # print(word)
        if word in lexicon:
            pinyin_list.append(lexicon[word])
        else:
            for text in word:
                pinyin_list.append(lexicon[text])

    # Insert "sil" between pinyin sequences
    return ' '.join(pinyin_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Chinese text to pinyin with 'sil' inserted between words.")
    parser.add_argument('input_text', type=str, help="The input Chinese text to convert.")
    parser.add_argument('--lexicon', type=str, default='lexicon_kaldi.txt', help="Path to the lexicon_kaldi.txt file.")

    # Parse the command-line arguments
    args = parser.parse_args()
    

    # Clean and split the input text
    processed_text = clean_and_split(args.input_text)

    print(f"Original text: {args.input_text}")
    print(f"Processed text: {processed_text}")

    # Path to your lexicon_kaldi.txt
    lexicon_path = 'lexicon_kaldi.txt'
    
    # Load the lexicon
    lexicon = load_lexicon(lexicon_path)


    # Convert text to pinyin
    pinyin_text = text_to_pinyin(processed_text, lexicon)

    # print(f"Original text: {input_text}")
    print("Pinyin text:", pinyin_text + ' sil')

