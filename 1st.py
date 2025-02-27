class FindS:
    def __init__(self, num_attributes):
        self.hypothesis = ['0'] * num_attributes 
    def fit(self, training_data):
        for example in training_data:
            # If the example is positive
            if example[-1] == 'yes':
                if self.hypothesis == ['0'] * len(example[:-1]):
                    self.hypothesis = example[:-1]
                else:
                    # Generalize the hypothesis
                    for i in range(len(self.hypothesis)):
                        if self.hypothesis[i] != example[i]:
                            self.hypothesis[i] = '?'  # Generalize to '?'
    
    def get_hypothesis(self):
        return self.hypothesis

# Example usage
if __name__ == "__main__":
    # Training data: Each row is an example, the last column is the target label
    training_data = [
        ['sunny', 'warm', 'high', 'false', 'yes'],
        ['sunny', 'warm', 'high', 'true', 'yes'],
        ['rainy', 'cold', 'high', 'false', 'no'],
        ['sunny', 'warm', 'high', 'false', 'yes'],
        ['sunny', 'warm', 'low', 'false', 'yes'],
        ['rainy', 'cold', 'high', 'true', 'no'],
        ['sunny', 'warm', 'low', 'true', 'yes'],
        ['rainy', 'warm', 'high', 'false', 'no'],
    ]

    # Create an instance of the FindS algorithm
    find_s = FindS(num_attributes=len(training_data[0]) - 1)

    # Fit the model to the training data
    find_s.fit(training_data)

    # Get the most specific hypothesis
    hypothesis = find_s.get_hypothesis()
    print("Most Specific Hypothesis:", hypothesis)