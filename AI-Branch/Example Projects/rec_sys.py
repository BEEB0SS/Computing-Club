from surprise import Dataset, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# Load the movielens-100k dataset
data = Dataset.load_builtin('ml-100k')

# Split into train and test set
trainset, testset = train_test_split(data, test_size=0.2)

# Use SVD algorithm
model = SVD()
model.fit(trainset)
predictions = model.test(testset)

# Calculate RMSE
accuracy = rmse(predictions)
print(f"RMSE of collaborative filtering: {accuracy:.2f}")
