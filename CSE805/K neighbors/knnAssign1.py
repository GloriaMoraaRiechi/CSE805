from collections import Counter
import pandas as pd
from prettytable import PrettyTable
from sklearn.neighbors import KNeighborsClassifier
#write the dataset
data = {
    "Name": ["Clarissa", "Maurice", "Jean", "Manry", "Xien", "Jeff", "Ana", "Betty", "Rashid"],
    "Weight": [51, 62, 69, 64, 65, 56, 58, 57, 55],
    "Height": [167, 182, 176, 173, 172, 174, 169, 173, 170],
    "Class": ["Underweight", "Normal", "Normal", "Normal", "Normal", "Underweight", "Underweight", "Normal", "Normal"]
}
# table = PrettyTable()
# table.field_names = ["Name", "Weight", "Height", "Class"]
# for i in range(len(data["Name"])):
#     table.add_row([data["Name"][i], data["Weight"][i], data["Height"][i], data["Class"][i]])
df = pd.DataFrame(data)
print(df)
X = df[["Weight", "Height"]]   #Features
y = df["Class"]                #target labels

knnModel = KNeighborsClassifier(n_neighbors=3, metric='manhattan')

knnModel.fit(X, y)

paul = pd.DataFrame([[57, 170]], columns=["Weight", "Height"])

predictedClass = knnModel.predict(paul)
print("Predicted class for paul is:", predictedClass[0])



#manual computation

#define a function to calculate the manhattan distance
def manhattan_distance(subject, point):     #tuples consisting two values each
    return abs(subject[0] - point[0]) + abs(subject[1] - point[1])

#function to perform KNN classification
def knn_classifier(subject1, data, k):
    distances = []

    #iterate over each data point in the dataset
    for i in range(len(data["Name"])):
        data_point = (data["Weight"][i], data["Height"])  #get the weight and height for each data point
        distance = manhattan_distance(
            (paul["Weight"], paul["Height"]),
            (data["Weight"][i], data["Height"][i])
        )
        distances.append((distance, data['Class'][i]))

        #sort the distances by distance value
        distances.sort(key=lambda  x: x[0])

        #get th nearest neighbours
        neighbors = distances[:k]

        # Count the classes of the k nearest neighbors
        classes = [neighbor[1] for neighbor in neighbors]
        majority_class = Counter(classes).most_common(1)[0][0]

        # Debug output to show distances and the selected neighbors
        print("Distances and classes of all individuals:")
        for dist, cls in distances:
            print(f"Distance: {dist}, Class: {cls}")

        print("\n5 Nearest Neighbors:")
        for neighbor in neighbors:
            print(f"Distance: {neighbor[0]}, Class: {neighbor[1]}")

        return majority_class

paul = {"Weight": 57, "Height": 1}
k = 5
predictedClass = knn_classifier(paul, data, k)
print(predictedClass)
