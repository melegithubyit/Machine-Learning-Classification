# Assuming the feature_list is a list of feature values [continent, season, wind speed, location, weather]
# and label_list is the temperature value
# Perform one-hot encoding for categorical features (continent, season, location, weather)


def featureize(feature_list, label_list):
    features_track = []
    for feature in feature_list:
        lst = []
        if feature in ['Asia', 'Europe', 'North America', 'South America']:
            for i in range(4):
                lst.append(int(feature == feature_list[0]))
            features_track += lst  # FOR THE CONTINENT
        elif feature in ['Spring', 'Summer', 'Autumn', 'Winter']:
            for i in range(4):
                lst.append(int(feature == feature_list[1]))
            features_track += lst  #  FOR THE SEASON
        elif feature in ['Location1', 'Location2', 'Location3', 'Location4']:
            for i in range(4):
                lst.append(int(feature == feature_list[3]))
            features_track += lst  # FOR THE LOCATION
        elif feature in ['Sunny', 'Cloudy', 'Rainy', 'Snowy']:
            for i in range(4):
                lst.append(int(feature == feature_list[4]))
            features_track += lst  # FOR THE WEATHER

    #  AS THE WIND VALUE IS ONLY TWO WE WILL APPEND IT AS IT IS
    transformed_features = features_track + [feature_list[2]]
    transformed_label = label_list

    return transformed_features, transformed_label



# OUR TRAINING_DATA_X AND TRAINING_DATA_Y ARE LISTS THAT WE GEET AFTER EXTRACTING OUR TRAINING DATA SET. THE TRAINING_DATA_X IS A LIST OF LISTS WHERE EACH INNER LIST REPRESENTS THE FEATURES FOR A SPECIFIC TRAINIGNG
def train(training_features, training_labels, laplace_smoothing):

    label_counts = {}
    for label in training_labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    # Calculate prior probabilities
    total_samples = len(training_labels)
    prior_probability = {}
    for label, count in label_counts.items():
        prior_probability[label] = count / total_samples

    # Count occurrences of each feature for each label
    feature_counts = {}
    feature_totals = {}
    for i, features in enumerate(training_features):
        label = training_labels[i]
        if label not in feature_counts:
            feature_counts[label] = {}
            feature_totals[label] = 0

        for feature in features:
            if feature not in feature_counts[label]:
                feature_counts[label][feature] = 1
            else:
                feature_counts[label][feature] += 1
            feature_totals[label] += 1

    # Calculate conditional probabilities with Laplace smoothing
    feature_probability = {}
    for label, features in feature_counts.items():
        feature_probability[label] = {}
        total_features = feature_totals[label]

        for feature, count in features.items():
            feature_probability[label][feature] = (count + laplace_smoothing) / (total_features + laplace_smoothing * total_features)

    # Return the model
    return prior_probability, feature_probability
