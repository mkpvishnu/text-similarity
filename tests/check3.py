import json
from datetime import datetime
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Initialize Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set your OpenAI API key
openai.api_key = ""

def extract_subject_temporal_info(original_text, summary_text):
    prompt = f"""
    Analyze and compare the temporal information in the following original text and its summary. 
    Original Text:{original_text}
    Summary Text:{summary_text}
    For each subject mentioned in either text, provide:
    1. The subject name
    2. Associated temporal expressions (both explicit and relative), normalized to a standard format. Use the most specific date available from either text.
    3. Actions or events related to the subject, ordered by their matching temporal expressions. Compare events from both texts and rephrase them consistently and meaningfully without missing context. If multiple events match a single temporal expression, combine them into one meaningful event. Ensure a strict 1:1 mapping between temporal expressions and events.
    4. The tense used for each action/event, normalized to past, present, or future.
    Format the output as two separate JSON objects, one for the original text and one for the summary. Each JSON should have subjects as keys, with values being objects containing 'temporal_expressions', 'events', and 'tenses' lists.
    Ensure that:
    - Temporal expressions are as specific as possible, preferring exact dates from the original text over relative expressions from the summary. But dont change the dates of summaries. Just compare and normalize if not in standard format.
    - Events are rephrased consistently between the original and summary texts.
    - The number of temporal expressions matches the number of events for each subject.
    Present the results as follows:
    Both original text analysis and summary text analysis as keys in json.
    "Original Text Analysis":[JSON object for original text],"Summary Text Analysis":[JSON object for summary text]
    Do not enclose the JSON objects in code view formatting.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI specialized in analyzing and comparing temporal information in texts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=2000
    )
    print(response.usage)
    print(response.choices[0].message.content)
    
    return json.loads(response.choices[0].message.content)

def analyze_temporal_shift(original_text, derived_text, derived_type, verbose=False):
    # Extract subject-based temporal information
    print(extract_subject_temporal_info(original_text, derived_text))

# Example usage
original_text = """
In the late 19th century, the race for technological innovation accelerated. Thomas Edison invented the phonograph in 1877, revolutionizing sound recording. By 1879, he had developed the first practical incandescent light bulb, illuminating homes across America by the early 1880s. Meanwhile, in Europe, Karl Benz was pioneering automobile technology, patenting the first gas-powered car in 1886. 

As the 20th century dawned, the Wright brothers achieved the first sustained, controlled, powered flight in 1903, marking the beginning of the aviation era. Just five years later, in 1908, Henry Ford introduced the Model T, making automobiles accessible to the average American. The following decades saw rapid advancements: in 1926, Robert Goddard launched the first liquid-fueled rocket, laying the groundwork for space exploration.

World War II (1939-1945) accelerated technological progress. The first electronic computer, ENIAC, was completed in 1945. In the post-war era, the Space Race began. The Soviet Union launched Sputnik 1 in 1957, and just four years later, in 1961, Yuri Gagarin became the first human in space. The United States responded, and on July 20, 1969, Neil Armstrong took his historic first step on the Moon.

In recent decades, the pace of innovation has only increased. The World Wide Web was invented by Tim Berners-Lee in 1989, transforming global communication. By the early 21st century, smartphones had become ubiquitous, with Apple's iPhone, introduced in 2007, leading the revolution. Looking to the future, companies like SpaceX, founded by Elon Musk in 2002, are now developing reusable rockets with the goal of making space travel more accessible and eventually colonizing Mars.
"""

summary_text = """
Technological innovation has rapidly evolved since the late 19th century. Edison's inventions of the phonograph (1877) and light bulb (1879) were followed by Benz's gas-powered car patent in 1886. The Wright brothers achieved powered flight in 1903, while Ford's Model T (1908) made cars widely accessible. World War II accelerated progress, leading to ENIAC, the first electronic computer, in 1945. The Space Race saw Gagarin reach space in 1961, followed by Armstrong's Moon landing in 1969. More recently, Berners-Lee's 1989 invention of the World Wide Web and the 2007 introduction of the iPhone have revolutionized communication. Looking ahead, SpaceX, founded in 2022, aims to make space travel more accessible and potentially colonize Mars.
"""

paraphrase_text = """
The first human lunar landing took place in the late 60s when Armstrong set foot on the Moon. 
NASA is looking ahead to Mars exploration in the future. 
Concurrently, Musk's company has been innovating rocket technology since the early 2000s, 
with the goal of making space travel more affordable and establishing a presence on the Red Planet.
"""

analyze_temporal_shift(original_text, summary_text, "summary")
