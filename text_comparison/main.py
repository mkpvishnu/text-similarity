# text_comparison/main.py

from comparison_system import TextComparisonSystem

def main():
    system = TextComparisonSystem()

    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "A fast auburn canine leaps above an idle hound."

    tasks = [
        ('summarization', text1, text2),
        ('similarity', text1, text2),
        ('paraphrase', text1, text2),
        ('contradiction', text1, text2),
    ]

    for task, t1, t2 in tasks:
        result = system.compare_texts(t1, t2, task)
        print(f"\nTask: {task}")
        print(f"Final Score: {result['final_score']:.4f}")
        print("Metrics:")
        for key, value in result.items():
            if key != 'final_score':
                print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()