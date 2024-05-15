from collections import defaultdict
import pandas as pd
import os
import sys


def train_bayes(train_data):
    # Initialize count numbers to 1
    count = {}
    prob = {}
    class_count = 0

    # Count the classes
    for index, instance in train_data.iterrows():
        class_count += 1
        y = instance["class"]
        if y not in count:
            count[y] = {"total": 1}
            prob[y] = {}
        else:
            count[y]["total"] += 1

        # Now we count the features for each class
        for feature in train_data.columns[1:]:
            xi = instance[feature]
            if feature not in count[y]:
                count[y][feature] = {}
                prob[y][feature] = {}
            if xi not in count[y][feature]:
                count[y][feature][xi] = 1
            else:
                count[y][feature][xi] += 1

    # Calculate probabilities
    for y in count:
        for feature in count[y]:
            if isinstance(count[y][feature], dict):
                total_count = sum(count[y][feature].values())  # Summing up values
                prob[y][feature] = {xi: count[y][feature][xi] / total_count for xi in count[y][feature]}
        
        # Calculate the total count for the class label
        total_feature_counts = sum(sum(count[y][feature].values()) for feature in count[y] if isinstance(count[y][feature], dict))
        total_instances = total_feature_counts + count[y]["total"]

        prob[y]["total"] = count[y]["total"] / total_instances

    return prob


def test_class(instance, y, prob):
    score = prob[y]['total'] / sum(prob[y]['total'] for y in prob)  # Initialize score with class probability
    
    # Calculate the score for each feature
    for feature, xi in instance.items():
        if feature in prob[y] and xi in prob[y][feature]:
            score *= prob[y][feature][xi]

    return score

def calculate_accuracy(predictions, actual_classes):
    correct_predictions = 0
    total_instances = len(predictions)
    
    # Iterate through each prediction and compare with the actual class
    for pred_class, actual_class in zip(predictions, actual_classes):
        if pred_class == actual_class:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_instances * 100
    return accuracy

def test_bayes(test_data, prob):
    predictions = []

    # Loop through each test instance
    for _, instance in test_data.iterrows():
        max_score = -1
        predicted_class = None

        # Loop through each class label
        for y in prob:
            # Calculate the score for the current class label
            score = test_class(instance, y, prob)
            
            # Update the predicted class label if the current score is higher
            if score > max_score:
                max_score = score
                predicted_class = y
        
        # Append the predicted class label to the predictions list
        predictions.append(predicted_class)

    return predictions


def generate_report(test_data, prob, output_file="report.txt"):
  """
  This function generates a report in text format containing:
  1. Conditional probabilities P(Xi = xi | Y = y)
  2. Class probabilities P(Y = y)
  3. Scores and predicted class for each test instance
  """

  # Open the output file for writing
  with open(output_file, 'w') as f:
      # Write report header information (e.g., title, date)
      f.write(f"Naive Bayes Classifier Report\nDate: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")

      # Write conditional probabilities
      f.write("**1. Conditional Probabilities (P(Xi = xi | Y = y))**\n")
      for y in prob:
          f.write(f"\nClass: {y}\n")
          for feature, feature_probs in prob[y].items():
              if isinstance(feature_probs, dict):
                  f.write(f"\tFeature: {feature}\n")
                  for xi, prob_xi in feature_probs.items():
                      f.write(f"\t\tP({feature} = {xi}) = {prob_xi:.4f}\n")

      # Write class probabilities
      f.write("\n**2. Class Probabilities (P(Y = y))**\n")
      for y, prob_y in prob.items():
          if isinstance(prob_y, dict):
              f.write(f"\tP(Y = {y}) = {prob_y['total']:.4f}\n")

      # Write test instance details
      f.write("\n**3. Test Instance Details**\n")
      f.write(f"\tInstance\tScore (no-recurrence)\tScore (recurrence)\tPredicted Class\n")
      for index, instance in test_data.iterrows():
          score_no_recurrence = test_class(instance, "no-recurrence-events", prob)
          score_recurrence = test_class(instance, "recurrence-events", prob)
          predicted_class = test_bayes(pd.DataFrame([instance]), prob)[0]
          f.write(f"\t{index+1}\t{score_no_recurrence:.4f}\t{score_recurrence:.4f}\t{predicted_class}\n")



if __name__ == "__main__":

    # Get the working dir
    current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    # Load in the training data file
    train_data = pd.read_csv(current_dir + "/data/breast-cancer-training.csv")
    del train_data[train_data.columns[0]]

    test_data = pd.read_csv(current_dir + "/data/breast-cancer-test.csv")
    del test_data[test_data.columns[0]]

    prob = train_bayes(train_data)

     # Test the classifier
    predictions = test_bayes(test_data, prob)
    
    # Calculate the accuracy
    actual_classes = test_data["class"]

    accuracy = calculate_accuracy(predictions, actual_classes)
    print("Accuracy:", accuracy, "%")
    generate_report(test_data, prob, output_file="sampleoutput.txt")