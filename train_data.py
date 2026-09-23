import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))

# Fix: filter out rows that aren't exactly 42 features (1 hand only)
data = []
labels = []
for d, l in zip(data_dict['data'], data_dict['labels']):
    if len(d) == 42:
        data.append(d)
        labels.append(l)

data = np.asarray(data)
labels = np.asarray(labels)

print(f"Loaded {len(data)} samples with {data.shape[1]} features")

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'{score*100:.2f}% of samples were predicted correctly!')

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved to model.p")