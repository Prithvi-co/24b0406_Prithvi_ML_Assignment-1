import pandas as pd
import numpy as np

def assign_grade(score):
    """Return grade based on score."""
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

def pandas_filter_pass(dataframe):
    """
    Returns a DataFrame with only records of students with grades A or B.

    Parameters:
        dataframe (pd.DataFrame): DataFrame containing student records

    Returns:
        pd.DataFrame: Filtered DataFrame with grades A or B
    """
    return dataframe[dataframe['Grade'].isin(['A', 'B'])]

def main():
    # Random seed for reproducibility
    np.random.seed(42)

    # Sample data
    names = ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Helen', 'Ivy', 'Jack']
    subjects = ['Math', 'Science', 'English', 'Math', 'Science', 'English', 'Math', 'Science', 'English', 'Math']
    scores = np.random.randint(50, 101, size=10)

    # Create DataFrame
    df = pd.DataFrame({
        'Name': names,
        'Subject': subjects,
        'Score': scores,
        'Grade': ''  # Initially empty
    })

    # Assign grades
    df['Grade'] = df['Score'].apply(assign_grade)

    # Print DataFrame sorted by Score descending
    print("Sorted DataFrame by Score (descending):\n")
    print(df.sort_values(by='Score', ascending=False), end="\n\n")

    # Average score per subject
    print("Average Score per Subject:\n")
    print(df.groupby('Subject')['Score'].mean(), end="\n\n")

    # Filter students with Grade A or B
    print("Students with Grade A or B:\n")
    print(pandas_filter_pass(df))

if __name__ == "__main__":
    main()
