import numpy as np

"""
q2.py

This program builds a classifier using naïve Bayes model for the data.
It predicts for new data based on the parameters calculated based on type of distribution of the feature values. 

@author: Anushree Sitaram Das (ad1707)
"""

def getData(filename):
    """
    load dataset from csv file
    :param filename:
    :return:
    """
    data = np.genfromtxt(filename, delimiter=",", skip_header=1,dtype=( bool, bool, bool, bool, bool, bool,int,int, bool),
                         names=["inHtml", "hasEmoji", "sentToList", "fromCom", "hasMyName", "hasSig", "sentences",
                                "words", "isSpam"])
    return data


def separate_data(dataset):
    """
    separate data according to class label(Y)

    :param dataset: dataset
    :return:        array of dataset separated based on their class
    """
    # initialize dictionary to store feature input rows based on output class label
    separated_classwise = dict()
    for i in range(len(dataset)):
        # get row from dataset
        row = dataset[i]
        # find out class
        class_value = row[-1]
        # if the class is not in the list separated_classwise
        # then add class
        if class_value not in separated_classwise:
            separated_classwise[class_value] = list()
        # append data to respective class
        separated_classwise[class_value].append(row)

    return separated_classwise


def calculate_likelihood_parameters(dataset):
    """
    Calculate likelihood parameters for each feature
    For discrete valued feature store P(feature = True | Y) and P(feature = False | Y)
    for all class labels Y
    For real valued feature store mu and sigma for gaussian distribution of that feature

    :param dataset: dataset
    :return:        parameters for all features
    """
    # total number of features
    no_features = len(dataset[0]) - 1

    # Find out index of features that are discrete
    is_discrete = [False for _ in range(no_features)]
    names = dataset.dtype.names
    for i in range(no_features):
        if len(np.unique(dataset[names[i]])) == 2:
            is_discrete[i] = True

    # separate data according to class labels
    separated = separate_data(dataset)

    # create dictionary to store all parameters
    all_feature_parameters = dict()

    for class_value, rows in separated.items():
        # index used to iterate through list is_discrete
        i = 0
        single_feature_parameters = []
        for column in zip(*rows):
            # calculate for all columns except last column which is class label 'isSpam'
            if i<no_features :
                if is_discrete[i]:
                    # if feature is discrete valued calculate P(feature = True|Y) and P(feature = False|Y) as follows:
                    # P(feature = True|Y) = (No of rows where feature == True & Y = class_value)/Total no of rows where Y = class_value
                    # P(feature = False|Y) = (No of rows where feature == False & Y = class_value)/Total no of rows where Y = class_value
                    sum_true = np.count_nonzero(column)
                    sum_false = len(column) - np.count_nonzero(column)
                    single_feature_parameters.append(['discrete',sum_true/len(column),sum_false/len(column)])
                else:
                    # if feature is discrete valued calculate mu = mean(feature values) and sigma = standard deviation(feature values)
                    # we can get P(feature = x| Y) from the gaussian distribution with given mu and sigma
                    single_feature_parameters.append(['real',np.mean(column), np.std(column)])
            i += 1
        # Store likelihoods for all features for each class label separately in the dictionary
        all_feature_parameters[class_value] = single_feature_parameters
    return all_feature_parameters


def class_probabilities(parameters, features,class_prob):
    """
    Calculate probability for each feature for each class label for given set of feature, i.e., P(feature|Y)
    Then calculate probability for each class label, i.e.,
    P(y|all features) = P(all features|y) / sum(P(all features|Y) for all values of Y)
    ,where P(all features|y) = P(feature1|y) * P(feature2|y) *..*P(featurek|y) * P(y)

    :param parameters:  parameters for all features
    :param features:    list of feature input
    :param class_prob:  Probabilities for each class label, i.e., [P(True), P(False)]
    :return:            Probablity
    """
    # create dictionary to store probability P(features|Y) for all values of Y
    final_class_prob = dict()
    # calculate numerator for P(y|all features)
    for class_value, class_parameters in parameters.items():
        # Initialize numerator = P(Y)
        if class_value == True:
            final_class_prob[class_value] = class_prob[0]
        else:
            final_class_prob[class_value] = class_prob[1]

        # Calculate P(feature|Y) for all features
        for i in range(len(class_parameters)):
            if class_parameters[i][0] == 'real':
                # if parameter is for real valued feature
                # use gaussian distribution with given mu and sigma to find probabilty
                mean = class_parameters[i][1]
                stdev = class_parameters[i][2]
                exponent = np.exp(-((features[i] - mean) ** 2 / (2 * stdev ** 2)))
                probability = min(1.0,((1 / (np.sqrt(2 * np.pi) * stdev)) * exponent))
            else:
                # if parameter is for discrete valued feature
                # get stored probabilty directly
                type, prob_true, prob_false = class_parameters[i]
                if features[i] == True:
                    probability = min(1.0,prob_true)
                else:
                    probability = min(1.0,prob_false)
            # multiply P(feature|Y) to numerator
            final_class_prob[class_value] *= probability


    denominator = 0
    # calculate denominator for P(y|all features)
    # summation of P(features|Y) for all Y
    for class_value,class_prob in final_class_prob.items():
        if class_value == True:
            denominator +=  (np.prod(class_prob))
        else:
            denominator += (np.prod(class_prob))

    # divide numerator with denominator to get final P(features|Y) for all Y
    for class_value, class_prob in final_class_prob.items():
        final_class_prob[class_value] /= denominator

    # return probability for each each class label
    return final_class_prob


def predict(parameters, features, class_prob):
    """
    Calculate probability for each each class label
    and return the label with maximum probability as class label

    :param parameters:  parameters for all features
    :param features:    list of feature input
    :param class_prob:  Probabilities for each class label, i.e., [P(True), P(False)]
    :return:
    """
    # get probability for each each class label
    probabilities = class_probabilities(parameters, features, class_prob)

    class_label = None
    best_probability = -1
    # find highest probability from probabilities for all classes
    # and assign that class to the current feature set
    for class_value, probability in probabilities.items():
        if class_label is None or probability > best_probability:
            best_probability = probability
            class_label = class_value

    # return class label and probability of that class label
    return class_label,best_probability

def classify_dataset(data_test,parameters,class_prob):
    """
    Classifies given dataset features and displays the model's error rate and accuracy

    :param parameters:  parameters for all features
    :param features:    list of feature input
    :param class_prob:  Probabilities for each class label, i.e., [P(True), P(False)]
    :return:            None
    """
    # actual class labels array
    y_test = data_test["isSpam"]
    m_test = len(y_test)
    # array of features
    X_test = np.column_stack((np.ones((m_test, 1)), data_test["inHtml"], data_test["hasEmoji"], data_test["sentToList"],
                              data_test["fromCom"], data_test["hasMyName"], data_test["hasSig"], data_test["sentences"],
                              data_test["words"]))

    # array that stores predictions
    pred = []

    # number of errors
    errors = 0
    for i in range(len(X_test)):
        # get prediction and probability for given features
        prediction, probability = predict(parameters, X_test[i, :],class_prob)
        pred.append(prediction)
        # if final probability is less than equal to 0.5 increment number of errors
        if round(probability,1) <= 0.5:
            errors += 1

    # calculate error rate as number of errors/total number of rows in training set
    error_rate = errors/len(X_test)
    print('Error Rate:',error_rate)

    # calculate accuracy
    sum = 0
    for i in range(len(pred)):
        if pred[i] == y_test[i]:
            sum += 1
    accuracy = sum / (float(len(y_test)))

    print("Accuracy:",accuracy * 100,"%")


def selected_classify_dataset(data_test,parameters,class_prob,features_index):
    """
    Classifies given dataset with selected features and displays the model's error rate and accuracy

    :param parameters:      parameters for all features
    :param features:        list of feature input
    :param class_prob:      Probabilities for each class label, i.e., [P(True), P(False)]
    :param features_index:  indices of features selected
    :return:                None
    """
    # class array
    y_test = data_test["isSpam"]
    # stores array of features to be processed
    X_test = []

    # find names of the features selected to use it to build inputs with selected features
    feature_name = []
    names = data_test.dtype.names
    for i in range(len(names)):
        if i in features_index:
            feature_name.append(names[i])

    # build array of inputs with selected features
    for name in feature_name:
        if len(X_test) == 0:
            X_test = np.array(data_test[name]).reshape(len(data_test[name]), 1)
        else:
            X_test = np.append(X_test, np.array(data_test[name]).reshape(len(data_test[name]), 1), 1)

    # get parameters for the selected features
    selected_feature_parameters = dict()
    for class_value, rows in parameters.items():
        single_feature_parameters = []
        for i in range(len(rows)):
            if i in features_index:
                single_feature_parameters.append(rows[i])

        selected_feature_parameters[class_value] = single_feature_parameters

    # array that stores predictions
    pred = []
    # number of errors
    errors = 0
    for i in range(len(X_test)):
        # get prediction and probability for given features
        prediction, probability = predict(selected_feature_parameters, X_test[i, :],class_prob)
        pred.append(prediction)
        # if final probability is less than equal to 0.5 increment number of errors
        if round(probability,1) <= 0.5:
            errors += 1

    # calculate error rate as number of errors/total number of rows in training set
    error_rate = errors/len(X_test)
    print('Error Rate:',error_rate)

    # calculate accuracy
    sum = 0
    for i in range(len(pred)):
        if pred[i] == y_test[i]:
            sum += 1
    accuracy = sum / (float(len(y_test)))

    print("Accuracy:",accuracy * 100,"%")


if __name__ == "__main__":
    # load training dataset
    data_train = getData("q3.csv")

    # Find number of rows with isSpam as True to calculate P(Y = True)
    sum_true = np.count_nonzero(data_train["isSpam"])
    # Find number of rows with isSpam as False to calculate P(Y = False)
    sum_false = len(data_train["isSpam"]) - np.count_nonzero(data_train["isSpam"])
    # Store P(Y = True) and P(Y = False) for calculating predictions later
    class_prob = [sum_true/len(data_train["isSpam"]),sum_false/len(data_train["isSpam"])]

    # call function that performs the parameter computation for the full feature set
    parameters = calculate_likelihood_parameters(data_train)

    # load testing dataset
    data_test = getData("q3b.csv")

    print("Classification for new data samples using the full feature set:")
    # call function that performs the classification for new data samples using the full feature set
    classify_dataset(data_test, parameters, class_prob)

    # chosen subset of the features
    features_index = [ 2, 3, 4, 6, 7]

    print("Classification based on the chosen subset of the features:")
    # call function that performs classification based on the chosen subset of the features
    selected_classify_dataset(data_test, parameters, class_prob, features_index)


