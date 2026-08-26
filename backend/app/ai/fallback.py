from typing import List, Dict

def get_fallback_tutor_response(question: str, level: str, sources: List[Dict[str, str]]) -> Dict:
    """
    Returns deterministic high-quality response when Gemini is unavailable or unconfigured.
    """
    q_lower = question.lower()
    
    if "function" in q_lower:
        explanation = "A Python function is a reusable block of code that executes when called. You can pass data (parameters) into it, and it can return data as a result."
        example = "def greet(name):\n    return f'Hello, {name}!'\n\nmessage = greet('Arjun')\nprint(message)  # Output: Hello, Arjun!"
        common_mistake = "Forgetting to write 'return', which causes Python to return None by default instead of your calculated result."
        practice_question = "Write a function named 'square' that takes a single number x as parameter and returns its square (x * x)."
    elif "join" in q_lower or "sql" in q_lower or "dbms" in q_lower:
        explanation = "SQL JOIN clauses are used to combine rows from two or more tables based on a related column between them. INNER JOIN requires a match in both tables, whereas LEFT JOIN includes all rows from the left table regardless of matches."
        example = "SELECT Students.name, Courses.title\nFROM Students\nLEFT JOIN Courses ON Students.course_id = Courses.id;"
        common_mistake = "Using INNER JOIN when you want to preserve students who have not enrolled in any course yet."
        practice_question = "If Table A has 5 rows and Table B has 3 matching rows, how many rows will INNER JOIN return versus LEFT JOIN?"
    else:
        explanation = f"In {level.lower()} programming concepts, breaking problems into modular functions and understanding data flow is fundamental to writing clean code."
        example = "# Variable declaration\ncount = 10\nif count > 5:\n    print('Count is high')"
        common_mistake = "Confusing assignment operator '=' with equality comparison operator '=='."
        practice_question = "What is the output of: print(5 == '5') in Python?"

    formatted_sources = []
    for s in sources:
        formatted_sources.append({
            "title": s.get("title", "Reference Material"),
            "snippet": s.get("snippet", s.get("text", ""))[:150]
        })

    if not formatted_sources:
        formatted_sources = [{
            "title": "Python & DBMS Fundamentals Reference",
            "snippet": "Functions encapsulate logic, and SQL JOINs relate tabular records."
        }]

    return {
        "explanation": explanation,
        "example": example,
        "common_mistake": common_mistake,
        "practice_question": practice_question,
        "sources": formatted_sources,
        "is_demo_mode": True
    }

def get_fallback_learning_plan(topic_name: str) -> List[Dict]:
    """Generates a deterministic 5-day intervention roadmap for weak topics."""
    return [
        {
            "day_number": 1,
            "title": f"{topic_name}: Core Concepts Review",
            "objective": f"Understand fundamental principles and syntax of {topic_name}.",
            "activity_type": "Guided Reading & Code Walkthrough",
            "resource_link": "https://docs.python.org/3/tutorial/"
        },
        {
            "day_number": 2,
            "title": f"{topic_name}: Interactive Practice",
            "objective": f"Solve basic exercises focused on {topic_name} execution flow.",
            "activity_type": "Targeted Code Exercises",
            "resource_link": "https://practice.edusaarthi.ai/exercises"
        },
        {
            "day_number": 3,
            "title": f"{topic_name}: Common Pitfalls & Debugging",
            "objective": "Identify and fix common logical errors and syntax mistakes.",
            "activity_type": "Debugging Lab",
            "resource_link": "https://practice.edusaarthi.ai/labs"
        },
        {
            "day_number": 4,
            "title": f"{topic_name}: Real-world Problem Solving",
            "objective": f"Apply {topic_name} to solve a scenario-based mini problem.",
            "activity_type": "Scenario Challenge",
            "resource_link": "https://practice.edusaarthi.ai/challenge"
        },
        {
            "day_number": 5,
            "title": f"{topic_name}: Targeted Reassessment",
            "objective": "Evaluate mastery improvement after completing intervention.",
            "activity_type": "5-Question Reassessment Quiz",
            "resource_link": "https://app.edusaarthi.ai/quizzes"
        }
    ]


def get_fallback_quiz(topic_name: str, num_questions: int = 5) -> List[Dict]:
    """Returns deterministic MCQ questions when Gemini is unavailable or unconfigured."""
    base_questions = [
        {
            "question_text": f"Which of the following best describes {topic_name}?",
            "option_a": "A high-level concept for structuring data",
            "option_b": "A low-level memory management technique",
            "option_c": "An operating system scheduling algorithm",
            "option_d": "A type of hardware interface",
            "correct_option": "A",
            "difficulty": "easy"
        },
        {
            "question_text": f"What is the primary purpose of {topic_name} in software development?",
            "option_a": "To slow down program execution",
            "option_b": "To improve code reuse and modularity",
            "option_c": "To increase the size of the codebase",
            "option_d": "To disable error handling",
            "correct_option": "B",
            "difficulty": "easy"
        },
        {
            "question_text": f"Which keyword is commonly associated with {topic_name} in Python?",
            "option_a": "class",
            "option_b": "import",
            "option_c": "def",
            "option_d": "All of the above depending on context",
            "correct_option": "D",
            "difficulty": "medium"
        },
        {
            "question_text": f"When implementing {topic_name}, which best practice should be followed?",
            "option_a": "Avoid writing any comments",
            "option_b": "Use meaningful variable names and keep functions short",
            "option_c": "Write all logic in a single function",
            "option_d": "Never test the code",
            "correct_option": "B",
            "difficulty": "medium"
        },
        {
            "question_text": f"Which of the following is a common mistake when learning {topic_name}?",
            "option_a": "Writing modular and readable code",
            "option_b": "Testing edge cases thoroughly",
            "option_c": "Confusing syntax with semantics",
            "option_d": "Using version control",
            "correct_option": "C",
            "difficulty": "hard"
        },
    ]
    return base_questions[:num_questions]
