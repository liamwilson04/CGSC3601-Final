import pandas as pd  # loading and preprocessing

import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# evaluation metrics, acc, cm, f1, (precision and recall if comparing models)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance # this might be removed

flag = True

func    = None
model   = None

while flag:
    func = input("Enter function (1: Models, 2: Weka): ")
    if (func.lower() == "1"):
        model = input("Enter model (SVM, RF): ")
        if (model.lower() == "svm" or model.lower() == "rf"):
            flag = False
        else:
            print("Invalid model. Please try again.")

    if func.lower() == "2":
        flag = False
    
    else:
        print("Invalid function. Please try again.")
    
        

#Load CSV and display some info
print("Loading CSV")
dataframe = pd.read_csv("full.csv") # df = dataframe

print(dataframe.isnull().sum())

dropped_cols = ["PassengerId", "Name", "Ticket", "Cabin", "Hometown", "Destination", "WikiId", "Age", "Lifeboat", "Body", "Embarked"]
dataframe = dataframe.drop(dropped_cols, axis=1) # drop all columns (irrelevant features) which are store in the array

print("Preprocessing data\n")
dataframe["Fare"] = dataframe["Fare"].fillna(dataframe["Fare"].median()) # median/mean for numerical values 
dataframe["Age_wiki"] = dataframe["Age_wiki"].fillna(dataframe["Age_wiki"].median()) 

labelEncoder = LabelEncoder()
dataframe["Sex"] = dataframe["Sex"].map({"male": 1, "female": 0}) # set the male and female values to numerical

if (func == "1"): 

    dataframe["Sex"] = dataframe["Sex"].map({"male": 1, "female": 0}) # set the male and female values to numerical

    def titles(row):
        if (pd.notna(row["Name_wiki"])):
            name = str(row["Name_wiki"]).lower()
            
            marriage_status = 0
            rank = 0
            nobility = 0
            
            for title in ["mrs", "mme"]:
                if title in name:
                    marriage_status = 1
                    break
            for title in ["col", "major", "capt", "dr", "rev", "father"]:
                if title in name:
                    rank = 1
                    break
            for title in ["jonkheer", "sir", "countess", "don"]:
                if title in name:
                    nobility = 1
                    break
            
            return pd.Series([marriage_status, rank, nobility])

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html
    dataframe[["Marriage_status", "Rank", "Nobility"]] = dataframe.apply(titles, axis=1)
    dataframe.pop('Name_wiki')

    dataframe = dataframe.dropna(subset=["Boarded"])
    dataframe = dataframe.dropna(subset=["Survived"])

    features = ["Pclass", "Sex", "Fare", "Age_wiki", "SibSp", "Parch", "Boarded", "Marriage_status", "Rank", "Nobility", "Youth"]
    features_train, features_test, target_train, target_test = train_test_split(dataframe[features], dataframe["Survived"], test_size=0.2, random_state=42)

    if (model.lower() == "svm"):
        scaler = StandardScaler()
        scaledFeaturesTrain = scaler.fit_transform(features_train)
        scaledFeaturesTest = scaler.transform(features_test)

        #\
        # Use metrics to evaluate the model overall
        # Theres also feature importance logic here, ignore this for now
        #/
        metrics = []
        modScores = []

        print("Training with all features")
        print("----------------------------------------------------------")
        svm = SVC() # initialization, not sure what the values other than kernel mean
        svm.fit(scaledFeaturesTrain, target_train) # fit the model based on the data
        prediction = svm.predict(scaledFeaturesTest) # make a prediction based on

        # All scores for model overall below, multiplied by 100 for percentage
        accuracy = accuracy_score(target_test, prediction)
        accuracy = accuracy * 100
        print(f"SVM accuracy: {accuracy:.2f}%")
        modScores.append(accuracy)
        metrics.append(f"Accuracy ({accuracy:.2f}%)")

        precision = precision_score(target_test, prediction)
        precision = precision * 100
        print(f"SVM precision: {precision:.2f}%")
        modScores.append(precision)
        metrics.append(f"Precision ({precision:.2f}%)")

        recall = recall_score(target_test, prediction)
        recall = recall * 100
        print(f"SVM recall: {recall:.2f}%")
        modScores.append(recall)
        metrics.append(f"Recall ({recall:.2f}%)")

        f1 = f1_score(target_test, prediction)
        f1 = f1 * 100
        print(f"SVM F1: {f1:.2f}%")
        modScores.append(f1)
        metrics.append(f"F1 Score ({f1:.2f}%)")

        cm = confusion_matrix(target_test, prediction)
        print("SVM Confusion Matrix", cm)

        # ROC goes here
        probabilities = svm.decision_function(scaledFeaturesTest)
        auc = roc_auc_score(target_test, probabilities)
        auc = auc * 100
        print(f"AUC score: {auc:.2f}%")
        modScores.append(auc)
        metrics.append(f"AUC Score ({auc:.2f}%)")

        plt.figure(figsize=(9, 6))
        plt.bar(metrics, modScores, color=["green", "red", "orange", "blue", "black", "yellow"])
        plt.title("SVM Overall Metrics")
        plt.xlabel("Metric")
        plt.ylabel("Score (%)")
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.show()

        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]

        plt.figure(figsize=(9, 6))
        plt.bar([f"True Negative ({tn})", f"False Positive ({fp})", f"False Negative ({fn})", f"True Positive ({tp})"], [tn, fp, fn, tp], color=["green", "red", "orange", "blue"])
        plt.title("SVM Confusion Matrix")
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()


        mic = mutual_info_classif(scaledFeaturesTrain, target_train)
        print("Mutual Info Classification scores")
        print(mic)

        plt.figure(figsize=(9, 6))
        plt.bar(features, mic, color="blue")
        plt.title("SVM Mutual Information Scores")
        plt.xlabel("Features")
        plt.ylabel("Raw MI Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print("----------------------------------------------------------\n")

        ############################################################################################################################################################################

        #\
        # Since the main goal is to see what features are important, use some metrics between each feature
        #/
        print("Training with individual features")
        print("----------------------------------------------------------")
        scaler = StandardScaler()
        corrScores = []
        predCounts = []

        for feature in features:
            print("Training with: ", feature)
            # correlation of the feature to Survived from -1 to 1
            # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
            pCorrelation = dataframe[feature].corr(dataframe["Survived"], method="pearson")
            print(f"Pearson correlation between {feature} and Survival: {pCorrelation:.2f}")
            corrScores.append(pCorrelation)

            feature_train = features_train[[feature]]
            feature_test = features_test[[feature]]

            scaledFeatureTrain = scaler.fit_transform(feature_train)
            scaledFeatureTest = scaler.transform(feature_test)

            svm = SVC()
            svm.fit(scaledFeatureTrain, target_train)

            prediction = svm.predict(scaledFeatureTest)
            print("Unique predictions:", np.unique(prediction, return_counts=True))

            accuracy = accuracy_score(target_test, prediction)
            acuracy = accuracy * 100
            print(f"SVM accuracy on {feature}: {accuracy:.2f}%")
            
            precision = precision_score(target_test, prediction)
            precision = precision * 100
            print(f"SVM precision on {feature}: {precision:.2f}%")

            recall = recall_score(target_test, prediction)
            recall = recall * 100
            print(f"SVM recall on {feature}: {recall:.2f}%")

            f1 = f1_score(target_test, prediction)
            f1 = f1 * 100
            print(f"SVM F1 on {feature}: {f1:.2f}%\n")
        print("----------------------------------------------------------")

        plt.figure(figsize=(9, 6))
        plt.bar(features, corrScores, color="orange")
        plt.title("SVM Pearson Correlation Scores on Individual Features")
        plt.xlabel("Features")
        plt.ylabel("Scores (-1 to 1)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    elif (model.lower() == "rf"):
        metrics = []
        modScores = []

        print("Training with Random Forest")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(features_train, target_train)
        rf_predictions = rf_model.predict(features_test)

        accuracy = accuracy_score(target_test, rf_predictions) * 100
        print(f"Random Forest accuracy: {accuracy:.2f}%")
        modScores.append(accuracy)
        metrics.append(f"Accuracy ({accuracy:.2f}%)")

        precision = precision_score(target_test, rf_predictions) * 100
        print(f"Random Forest precision: {precision:.2f}%")
        modScores.append(precision)
        metrics.append(f"Precision ({precision:.2f}%)")

        recall = recall_score(target_test, rf_predictions) * 100
        print(f"Random Forest recall: {recall:.2f}%")
        modScores.append(recall)
        metrics.append(f"Recall ({recall:.2f}%)")

        f1 = f1_score(target_test, rf_predictions) * 100
        print(f"Random Forest F1: {f1:.2f}%")
        modScores.append(f1)
        metrics.append(f"F1 Score ({f1:.2f}%)")

        rf_cm = confusion_matrix(target_test, rf_predictions)
        print(f"Random Forest confusion matrix:\n{rf_cm}\n")

        probabilities = rf_model.predict_proba(features_test)[:, 1]
        auc = roc_auc_score(target_test, probabilities)
        auc = auc * 100
        print(f"AUC score: {auc:.2f}%")
        modScores.append(auc)
        metrics.append(f"AUC Score ({auc:.2f}%)")

        plt.figure(figsize=(9, 6))
        plt.bar(metrics, modScores, color=["green", "red", "orange", "blue", "black", "yellow"])
        plt.title("RF Overall Metrics without Highest Gini Scored Features")
        plt.xlabel("Metric")
        plt.ylabel("Score (%)")
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.show()


        # Gini-based feature importances
        importances = rf_model.feature_importances_
        print(f"Random Forest Feature Importances (Gini-based): {importances}\n")

        plt.figure(figsize=(9, 6))
        plt.bar(features, importances, color="blue")
        plt.title("RF Gini Importance Scores")
        plt.xlabel("Features")
        plt.ylabel("Raw Gini Score (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        ###################################################################################################################

        print("Training SVM on individual features")
        for feature in features:
            print("Training with:", feature)

            correlation = dataframe[feature].corr(dataframe["Survived"])
            print(f"Correlation between {feature} and Survival: {correlation:.2f}")

            feature_train = features_train[[feature]]
            feature_test = features_test[[feature]]

            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(feature_train, target_train)
            rf_predictions = rf_model.predict(feature_test)

            rf_accuracy = accuracy_score(target_test, rf_predictions) * 100
            print(f"Random Forest accuracy: {rf_accuracy:.2f}%")

            rf_precision = precision_score(target_test, rf_predictions) * 100
            print(f"Random Forest precision: {rf_precision:.2f}%")

            rf_recall = recall_score(target_test, rf_predictions) * 100
            print(f"Random Forest recall: {rf_recall:.2f}%")

            rf_f1 = f1_score(target_test, rf_predictions) * 100
            print(f"Random Forest F1: {rf_f1:.2f}%")

            rf_cm = confusion_matrix(target_test, rf_predictions)
            print(f"Random Forest confusion matrix:\n{rf_cm}\n")

elif (func == "2"):

    dataframe = dataframe.dropna(subset=["Survived"])
    dataframe = dataframe.dropna(subset=["Boarded"])

    def titles(row):
        if (pd.notna(row["Name_wiki"])):
            name = str(row["Name_wiki"]).lower()
            
            marriage_status = "No"
            rank = "No"
            nobility = "No"
            
            for title in ["mrs", "mme"]:
                if title in name:
                    marriage_status = "Yes"
                    break
            for title in ["col", "major", "capt", "dr", "rev", "father"]:
                if title in name:
                    rank = "Yes"
                    break
            for title in ["jonkheer", "sir", "countess", "don"]:
                if title in name:
                    nobility = "Yes"
                    break
            
            return pd.Series([marriage_status, rank, nobility])

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html
    dataframe[["Marriage_status", "Rank", "Nobility"]] = dataframe.apply(titles, axis=1)
    dataframe.pop('Name_wiki')

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html
    dataframe.loc[dataframe["Age_wiki"] <= 17, "Youth"] = "Yes"
    dataframe.loc[dataframe["Age_wiki"] > 17, "Youth"] = "No"

    surv = dataframe.pop('Survived')
    dataframe['Survived'] = surv

    dataframe["Boarded"] = dataframe["Boarded"].map({"Cherbourg": 3, "Southampton": 2, "Queenstown": 1})
    dataframe["Survived"] = dataframe["Survived"].map({1: "Yes", 0: "No"})

    dataframe.to_csv("preprocessed_titanic.csv", index=False)