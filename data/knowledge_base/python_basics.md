# Python Fundamentals

## Variables and Data Types
In Python, variables are created when you assign a value to them. Python has dynamic typing:
- Integers (`int`): Whole numbers like 10, -5, 0.
- Floats (`float`): Decimal numbers like 3.14, 0.0.
- Strings (`str`): Text enclosed in quotes like "hello".
- Booleans (`bool`): `True` or `False`.

## Functions
A function is a block of organized, reusable code that performs a single action:
- Defined using the `def` keyword.
- Parameters pass inputs into functions.
- The `return` keyword sends results back to the caller.

Common mistake: Forgetting to return a value, causing the function to implicitly return `None`.

Example:
```python
def add_numbers(a, b):
    return a + b
```

## Control Flow: Conditionals and Loops
Conditionals use `if`, `elif`, and `else` to execute code based on boolean expressions.
Loops repeat actions:
- `for` loops iterate over sequences (lists, ranges).
- `while` loops run as long as a condition remains `True`.

Common mistake: Creating an infinite `while` loop by omitting the loop variable increment.
