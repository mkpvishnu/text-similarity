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

def extract_subject_temporal_info(text):
    prompt = f"""
    Extract subject-based temporal information from the following text. 
    For each subject mentioned, provide:
    1. The subject name
    2. Associated temporal expressions (both explicit and relative)
    3. Actions or events related to the subject with their temporal context
    4. The tense used for each action/event
    
    Format the output as a JSON where each key is a subject, and the value is an object with keys: 'temporal_expressions', 'events', 'tenses' and dont enclose the json object in json code view.
    
    Text: {text}
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts temporal information from text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=2000
    )
    print(response.choices[0].message.content)
    
    return json.loads(response.choices[0].message.content)

def calculate_temporal_preservation_score(original_info, derived_info, derived_type):
    scores = {}
    
    # 1. Key Temporal Information Preservation (0-30 points)
    key_temporal_events = extract_key_temporal_events(original_info)
    preserved_events = [event for event in key_temporal_events if event_is_preserved(event, derived_info)]
    scores['key_info_preservation'] = len(preserved_events) / len(key_temporal_events) * 30 if key_temporal_events else 30
    
    # 2. Subject-Time Association Preservation (0-25 points)
    subject_time_scores = []
    for subject, info in original_info.items():
        if subject_is_preserved(subject, derived_info):
            matched_subject = find_matching_subject(subject, derived_info)
            original_times = set(info['temporal_expressions'])
            derived_times = set(derived_info[matched_subject]['temporal_expressions'])
            preservation = max(temporal_expression_similarity(ot, dt) for ot in original_times for dt in derived_times)
            subject_time_scores.append(preservation)
    scores['subject_time_preservation'] = np.mean(subject_time_scores) * 25 if subject_time_scores else 25
    
    # 3. Temporal Sequence Accuracy (0-20 points)
    original_sequence = extract_temporal_sequence(original_info)
    derived_sequence = extract_temporal_sequence(derived_info)
    sequence_accuracy = improved_sequence_similarity(original_sequence, derived_sequence)
    scores['temporal_sequence_accuracy'] = sequence_accuracy * 20
    
    # 4. Tense Consistency (0-15 points)
    original_tenses = [tense for info in original_info.values() for tense in info['tenses']]
    derived_tenses = [tense for info in derived_info.values() for tense in info['tenses']]
    tense_consistency = len(set(original_tenses).intersection(set(derived_tenses))) / len(set(original_tenses)) if original_tenses else 1
    scores['tense_consistency'] = tense_consistency * 15
    
    # 5. Temporal Focus Retention (0-10 points)
    original_focus = analyze_temporal_focus(original_info)
    derived_focus = analyze_temporal_focus(derived_info)
    focus_retention = 1 - min(abs(original_focus - derived_focus) / 365, 1)  # Normalize to [0, 1]
    scores['focus_retention'] = focus_retention * 10
    
    # Calculate total score
    total_score = sum(scores.values())
    
    return {
        'total_score': round(total_score, 2),
        'component_scores': {k: round(v, 2) for k, v in scores.items()}
    }

def extract_key_temporal_events(info):
    return [(subject, event, time) 
            for subject, data in info.items() 
            for event, time in zip(data['events'], data['temporal_expressions'])]

def event_is_preserved(event, derived_info):
    subject, event_desc, time = event
    for derived_subject, data in derived_info.items():
        event_similarity = max(semantic_similarity(event_desc, de) for de in data['events'])
        time_similarity = max(temporal_expression_similarity(time, dt) for dt in data['temporal_expressions'])
        if event_similarity > 0.6 and time_similarity > 0.6:
            return True
    return False

def subject_is_preserved(subject, derived_info):
    return any(semantic_similarity(subject, ds) > 0.6 for ds in derived_info.keys())

def find_matching_subject(subject, derived_info):
    return max(derived_info.keys(), key=lambda ds: semantic_similarity(subject, ds))

def semantic_similarity(s1, s2):
    embeddings = model.encode([s1, s2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def temporal_expression_similarity(t1, t2):
    if is_explicit_date(t1) and is_explicit_date(t2):
        return date_similarity(t1, t2)
    return semantic_similarity(t1, t2)

def date_similarity(d1, d2):
    try:
        date1 = datetime.strptime(d1, "%B %d, %Y")
        date2 = datetime.strptime(d2, "%B %Y")
        return 1 if date1.year == date2.year and date1.month == date2.month else 0
    except ValueError:
        return semantic_similarity(d1, d2)

def is_explicit_date(time_expr):
    return any(word in time_expr.lower() for word in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']) or any(char.isdigit() for char in time_expr)

def extract_temporal_sequence(info):
    events = [(subject, event, time) for subject, data in info.items() for event, time in zip(data['events'], data['temporal_expressions'])]
    return sorted(events, key=lambda x: normalize_temporal_expressions(x[2]))

def normalize_temporal_expressions(expression):
    base_date = datetime(2024, 7, 9)  # Current date as per the system's knowledge
    
    try:
        return (datetime.strptime(expression, "%Y-%m-%d") - base_date).days
    except:
        relative_map = {
            "past": -365, "present": 0, "future": 365,
            "recent past": -30, "near future": 30,
            "last year": -365, "next year": 365,
            "last month": -30, "next month": 30,
            "yesterday": -1, "tomorrow": 1
        }
        return relative_map.get(expression.lower(), 0)

def improved_sequence_similarity(seq1, seq2):
    total_similarity = 0
    for event1 in seq1:
        best_match = max(seq2, key=lambda event2: event_similarity(event1, event2))
        total_similarity += event_similarity(event1, best_match)
    return total_similarity / len(seq1) if seq1 else 0

def event_similarity(event1, event2):
    subject_sim = semantic_similarity(event1[0], event2[0])
    event_sim = semantic_similarity(event1[1], event2[1])
    time_sim = temporal_expression_similarity(event1[2], event2[2])
    return (subject_sim + event_sim + time_sim) / 3

def analyze_temporal_focus(info):
    all_expressions = [expr for subject_info in info.values() for expr in subject_info['temporal_expressions']]
    normalized = [normalize_temporal_expressions(expr) for expr in all_expressions]
    return sum(normalized) / len(normalized) if normalized else 0

def analyze_temporal_shift(original_text, derived_text, derived_type):
    # Extract subject-based temporal information
    original_info = extract_subject_temporal_info(original_text)
    derived_info = extract_subject_temporal_info(derived_text)
    
    # Calculate temporal preservation score
    preservation_score = calculate_temporal_preservation_score(original_info, derived_info, derived_type)
    
    return preservation_score

# Example usage
original_text = """
In the late 19th century, the race for technological innovation accelerated. Thomas Edison invented the phonograph in 1877, revolutionizing sound recording. By 1879, he had developed the first practical incandescent light bulb, illuminating homes across America by the early 1880s. Meanwhile, in Europe, Karl Benz was pioneering automobile technology, patenting the first gas-powered car in 1886. 

As the 20th century dawned, the Wright brothers achieved the first sustained, controlled, powered flight in 1903, marking the beginning of the aviation era. Just five years later, in 1908, Henry Ford introduced the Model T, making automobiles accessible to the average American. The following decades saw rapid advancements: in 1926, Robert Goddard launched the first liquid-fueled rocket, laying the groundwork for space exploration.

World War II (1939-1945) accelerated technological progress. The first electronic computer, ENIAC, was completed in 1945. In the post-war era, the Space Race began. The Soviet Union launched Sputnik 1 in 1957, and just four years later, in 1961, Yuri Gagarin became the first human in space. The United States responded, and on July 20, 1969, Neil Armstrong took his historic first step on the Moon.

In recent decades, the pace of innovation has only increased. The World Wide Web was invented by Tim Berners-Lee in 1989, transforming global communication. By the early 21st century, smartphones had become ubiquitous, with Apple's iPhone, introduced in 2007, leading the revolution. Looking to the future, companies like SpaceX, founded by Elon Musk in 2002, are now developing reusable rockets with the goal of making space travel more accessible and eventually colonizing Mars.
"""

summary_text = """
Technological innovation has rapidly evolved since the late 19th century. Edison's inventions of the phonograph (1877) and light bulb (1879) were followed by Benz's gas-powered car patent in 1886. The Wright brothers achieved powered flight in 1903, while Ford's Model T (1908) made cars widely accessible. World War II accelerated progress, leading to ENIAC, the first electronic computer, in 1945. The Space Race saw Gagarin reach space in 1961, followed by Armstrong's Moon landing in 1969. More recently, Berners-Lee's 1989 invention of the World Wide Web and the 2007 introduction of the iPhone have revolutionized communication. Looking ahead, SpaceX, founded in 2002, aims to make space travel more accessible and potentially colonize Mars.
"""

paraphrase_text = """
The first human lunar landing took place in the late 60s when Armstrong set foot on the Moon. 
NASA is looking ahead to Mars exploration in the future. 
Concurrently, Musk's company has been innovating rocket technology since the early 2000s, 
with the goal of making space travel more affordable and establishing a presence on the Red Planet.
"""

summary_score = analyze_temporal_shift(original_text, summary_text, "summary")

print("Summary Temporal Information Preservation Score:")
print(json.dumps(summary_score, indent=2))