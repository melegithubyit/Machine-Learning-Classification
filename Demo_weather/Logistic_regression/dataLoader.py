
#  THIS FUNCTION TAKES THE FILE AND READ IT LINE BY LINE AND STORE THE SPLITTED WORDS IN A LIST CALLED DATA
def load_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        line = line.strip()
        if line:
            data.append(line.split(','))
    return data


# IN THIS FUNCTION WE DID REDUNDANCY CHECK UP
def clear_data(data):
    mappings = {}
    temperatures = []
    for i in range(len(data[0])):
        if i in [0, 1, 2, 3, 4]:
            vals = []
            for row in data:
                vals.append(row[i])
            unique_values = list(set(vals))
            mappings[i] = {value: index for index, value in enumerate(unique_values)}

    for row in data:
        temperatures.append(row[-1])
    unique_temperatures = list(set(temperatures))
    unique_temperatures = list(set([row[-1] for row in data]))
    temperature_mapping = {value: index for index, value in enumerate(unique_temperatures)}

    for row in data:
        for i in range(len(row)-1):
            if i in [0, 1, 2, 3, 4]:
                row[i] = mappings[i][row[i]]
            else:
                row[i] = float(row[i])
        row[-1] = temperature_mapping[row[-1]]

    return data
