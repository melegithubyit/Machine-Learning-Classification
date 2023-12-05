import csv


def load_data(file_path):
    features, label = [], []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            features.append(row[:-1])
            label.append(row[-1])
            # LABEL IS A LIST OF NORMAL, HOT AND COLD
            #  FEATURES IS A LIST OF CONTINENT, SEASON, WINDSPEED, LOCATION, WEATHER
    return features, label
