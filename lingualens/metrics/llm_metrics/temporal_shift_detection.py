import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Initialize Sentence-BERT model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Set your OpenAI API key
def response_from_llm(llm, llm_apikey, prompt):
        openai.api_key = llm_apikey
        if llm in ["gpt-3.5-turbo-0125", "gpt-4", "gpt-4o"]:
            response = openai.ChatCompletion.create(
            model=llm,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts temporal information from text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        return response
     

def generate_gpt_feedback(original_text, summary_text, scores):
    prompt = f"""

Original Text:
{original_text}

Summary:
{summary_text}

Scores:
Total Score: {scores['total_score']}/100
Key Information Preservation: {scores['component_scores']['key_info_preservation']}/30
Subject-Time Association Preservation: {scores['component_scores']['subject_time_preservation']}/25
Temporal Sequence Accuracy: {scores['component_scores']['temporal_coherence']}/20
Tense Consistency: {scores['component_scores']['tense_consistency']}/10
Temporal Focus Retention: {scores['component_scores']['focus_retention']}/10
Information Condensation Quality: {scores['component_scores']['condensation_quality']}/5

Please provide:
1. An overall assessment of the summary's quality in preserving temporal information, while briefly fly through the positives, highlight the shortcoming precisly.

Your response should be short, insightful, and actionable.
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI specialized in analyzing temporal information in texts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    return response.choices[0].message['content']

def extract_subject_temporal_info(original_text, summary_text, llm, llm_apikey):
    prompt = f"""
     Analyze and compare the temporal information in the following original text and its summary.
    Original Text: {original_text}
    Summary Text: {summary_text}
    For each subject mentioned in either text, provide:

    1. The subject name.
    2. Associated temporal expressions (both explicit and relative), normalized to a standard format. Use the most specific date available from either text.
    3. Actions or events related to the subject, ordered by their matching temporal expressions. Compare events from both texts and meaningfully rephrase the events in summary wrt to original text if they are same events.But dont add new events in summary text. Only rephrase exisiting ones. If multiple events match a single temporal expression, combine them into one meaningful event. Ensure a strict 1:1 mapping between temporal expressions and events.
    4. The tense used for each action/event, normalized to past, present, or future.
    Special Instructions:

    Pay close attention to details in the summary text, ensuring that specific information such as "Wednesday was a day off" is captured accurately.
    Temporal expressions should be as specific as possible, preferring exact dates from the original text over relative expressions from the summary if they refer to the same event.
    Strictly Do not change the specific dates or add events in the summary. Only normalize and compare events that are mentioned in both texts.
    Format the output as two separate JSON objects, one for the original text and one for the summary. Each JSON should have subjects as keys, with values being objects containing 'temporal_expressions', 'events', and 'tenses' lists. The number of temporal expressions should match the number of events for each subject.

    Present the results as follows:
    Both original text analysis and summary text analysis as keys in JSON.
    "Original Text Analysis": [JSON object for original text], "Summary Text Analysis": [JSON object for summary text].
    Do not enclose the JSON objects in code view formatting.
    """
    response = response_from_llm(llm, llm_apikey, prompt)
           
    print(response.usage)
    print(response.choices[0].message.content)
    
    return json.loads(response.choices[0].message.content)

def semantic_similarity(s1, s2):
    embeddings = model.encode([s1, s2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def improved_subject_matching(subject, derived_info):
    best_match = max(derived_info.keys(), key=lambda ds: semantic_similarity(subject, ds))
    if semantic_similarity(subject, best_match) > 0.5:
        return best_match
    return None

def improved_event_is_preserved(event, derived_info):
    subject, event_desc, time = event
    for derived_subject, data in derived_info.items():
        event_similarity = max(semantic_similarity(event_desc, de) for de in data['events'])
        time_similarity = max(semantic_similarity(time, dt) for dt in data['temporal_expressions'])
        if event_similarity > 0.5 or time_similarity > 0.8:
            return True
    return False

def calculate_temporal_preservation_score(original_info, derived_info):
    scores = {}
    
    # 1. Key Temporal Information Preservation (0-30 points)
    key_temporal_events = extract_key_temporal_events(original_info)
    preserved_events = [event for event in key_temporal_events if improved_event_is_preserved(event, derived_info)]
    scores['key_info_preservation'] = len(preserved_events) / len(key_temporal_events) * 30 if key_temporal_events else 35
    
    # 2. Subject-Time Association Preservation (0-20 points)
    subject_time_scores = []
    for subject, info in original_info.items():
        matched_subject = improved_subject_matching(subject, derived_info)
        if matched_subject:
            original_times = set(info['temporal_expressions'])
            derived_times = set(derived_info[matched_subject]['temporal_expressions'])
            preservation = max(semantic_similarity(ot, dt) for ot in original_times for dt in derived_times)
            subject_time_scores.append(preservation)
    scores['subject_time_preservation'] = np.mean(subject_time_scores) * 25 if subject_time_scores else 30
    
    # 3. Temporal Sequence Accuracy (0-20 points)
    original_sequence = extract_temporal_sequence(original_info)
    derived_sequence = extract_temporal_sequence(derived_info)
    temporal_coherence = improved_sequence_similarity(original_sequence, derived_sequence)
    scores['temporal_coherence'] = temporal_coherence * 20
    
    # 4. Tense Consistency (0-10 points)
    tense_consistency = calculate_tense_consistency(original_info, derived_info)
    scores['tense_consistency'] = tense_consistency * 10
    
    # 5. Information Condensation Quality (0-10 points)
    condensation_quality = evaluate_condensation_quality(original_info, derived_info)
    scores['condensation_quality'] = condensation_quality * 5
    
    # Calculate total score
    total_score = sum(scores.values())
    
    return {
        'total_score': round(total_score, 2),
        'component_scores': {k: round(v, 2) for k, v in scores.items()}
    }
    
def calculate_tense_consistency(original_info, derived_info):
    original_tenses = set(tense for info in original_info.values() for tense in info['tenses'])
    derived_tenses = set(tense for info in derived_info.values() for tense in info['tenses'])
    
    # Convert tenses to a standard format
    tense_mapping = {
        'past': 'past',
        'present': 'present',
        'future': 'future',
        'past simple': 'past',
        'present simple': 'present',
        'present perfect': 'present'
    }
    
    original_tenses = {tense_mapping.get(t, t) for t in original_tenses}
    derived_tenses = {tense_mapping.get(t, t) for t in derived_tenses}
    
    consistency = len(original_tenses.intersection(derived_tenses)) / len(original_tenses) if original_tenses else 1
    return consistency

def extract_key_temporal_events(info):
    return [(subject, event, time) 
            for subject, data in info.items() 
            for event, time in zip(data['events'], data['temporal_expressions'])]

def extract_temporal_sequence(info):
    events = []
    for subject, data in info.items():
        for event, time in zip(data['events'], data['temporal_expressions']):
            events.append((subject, event, time))
    return events

def improved_sequence_similarity(original_sequence, derived_sequence):
    total_similarity = 0
    matched_count = 0
    
    for derived_subject, derived_event, derived_time in derived_sequence:
        # Find the best match based on event similarity
        best_match = max(original_sequence, key=lambda x: semantic_similarity(derived_event, x[1]))
        original_subject, original_event, original_time = best_match
        
        # Calculate similarities
        event_similarity = semantic_similarity(derived_event, original_event)
        time_similarity = semantic_similarity(derived_time, original_time)
        
        # Print details for debugging
        print(f"Derived Subject: {derived_subject}, Derived Event: {derived_event}, Derived Time: {derived_time}")
        print(f"Best Match - Original Subject: {original_subject}, Original Event: {original_event}, Original Time: {original_time}")
        print(f"Event Similarity: {event_similarity}, Time Similarity: {time_similarity}")
        
        # Check if the match is good enough
        if event_similarity > 0.5:  # We only check event similarity threshold
            total_similarity += time_similarity
            matched_count += 1
            print(f"Match found with similarity score: {time_similarity}")
        else:
            print("No sufficient event similarity, skipping match.")
        
        print()  # Blank line for readability
    
    # Calculate the final score
    final_score = total_similarity / len(derived_sequence) if derived_sequence else 0
    print(f"Final Similarity Score: {final_score}")
    return final_score


def event_similarity(event1, event2):
    subject_sim = semantic_similarity(event1[0], event2[0])
    event_sim = semantic_similarity(event1[1], event2[1])
    return (subject_sim + event_sim) / 2


def evaluate_condensation_quality(original_info, derived_info):
    original_event_count = sum(len(data['events']) for data in original_info.values())
    derived_event_count = sum(len(data['events']) for data in derived_info.values())
    
    # Calculate the ratio of preserved information
    preservation_ratio = derived_event_count / original_event_count if original_event_count > 0 else 1
    
    # Penalize if the summary is too long or too short
    if preservation_ratio > 0.8:
        return 0.8  # Penalize for not condensing enough
    elif preservation_ratio < 0.2:
        return 0.5  # Penalize for condensing too much
    else:
        return 1 - abs(0.5 - preservation_ratio)  # Optimal condensation around 50%

def analyze_temporal_shift_with_llm(original_text, derived_text,llm, llm_apikey, verbose=False):
    # Extract subject-based temporal information
    llm_response = extract_subject_temporal_info(original_text, derived_text, llm, llm_apikey)
    original_info = llm_response["Original Text Analysis"]
    derived_info = llm_response["Summary Text Analysis"]
    
    # Calculate temporal preservation score
    preservation_score = calculate_temporal_preservation_score(original_info, derived_info)
    
    if verbose:
        gpt_feedback = generate_gpt_feedback(original_text, derived_text, preservation_score)
        preservation_score['gpt_feedback'] = gpt_feedback
    
    return preservation_score

# Example usage
# original_text = """
# In the late 19th century, the race for technological innovation accelerated. Thomas Edison invented the phonograph in 1877, revolutionizing sound recording. By 1879, he had developed the first practical incandescent light bulb, illuminating homes across America by the early 1880s. Meanwhile, in Europe, Karl Benz was pioneering automobile technology, patenting the first gas-powered car in 1886. 

# As the 20th century dawned, the Wright brothers achieved the first sustained, controlled, powered flight in 1903, marking the beginning of the aviation era. Just five years later, in 1908, Henry Ford introduced the Model T, making automobiles accessible to the average American. The following decades saw rapid advancements: in 1926, Robert Goddard launched the first liquid-fueled rocket, laying the groundwork for space exploration.

# World War II (1939-1945) accelerated technological progress. The first electronic computer, ENIAC, was completed in 1945. In the post-war era, the Space Race began. The Soviet Union launched Sputnik 1 in 1957, and just four years later, in 1961, Yuri Gagarin became the first human in space. The United States responded, and on July 20, 1969, Neil Armstrong took his historic first step on the Moon.

# In recent decades, the pace of innovation has only increased. The World Wide Web was invented by Tim Berners-Lee in 1989, transforming global communication. By the early 21st century, smartphones had become ubiquitous, with Apple's iPhone, introduced in 2007, leading the revolution. Looking to the future, companies like SpaceX, founded by Elon Musk in 2002, are now developing reusable rockets with the goal of making space travel more accessible and eventually colonizing Mars.
# """

# summary_text = """
# Technological innovation has rapidly evolved since the late 19th century. Edison's inventions of the phonograph (1877) and light bulb (1879) were followed by Benz's gas-powered car patent in 1886. The Wright brothers achieved powered flight in 1903, while Ford's Model T (1908) made cars widely accessible. World War II accelerated progress, leading to ENIAC, the first electronic computer, in 1945. The Space Race saw Gagarin reach space in 1961, followed by Armstrong's Moon landing in 1969. More recently, Berners-Lee's 1989 invention of the World Wide Web and the 2007 introduction of the iPhone have revolutionized communication. Looking ahead, SpaceX, founded in 2002, aims to make space travel more accessible and potentially colonize Mars.
# """

# paraphrase_text = """
# The first human lunar landing took place in the late 60s when Armstrong set foot on the Moon. 
# NASA is looking ahead to Mars exploration in the future. 
# Concurrently, Musk's company has been innovating rocket technology since the early 2000s, 
# with the goal of making space travel more affordable and establishing a presence on the Red Planet.
# """

# summary_score = analyze_temporal_shift_with_llm(original_text, summary_text, verbose=False)

# print("Summary Temporal Information Preservation Score:")
# print(json.dumps(summary_score, indent=2))