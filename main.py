#2) email spam detection
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

email=["click here to earn money","tommarrow is holiday","click to get bonus","click to download image"]
spams=np.array([1,0,1,1])
#CountVectorizer=it is used to convert text into the numerial value
vectorizer=CountVectorizer().fit(email)
s=vectorizer.transform(email) #after convert text into value than it can be stored in email list
#model=it is used to train the logisticRegression to identify the spam text
model=LogisticRegression().fit(s,spams)
print(model)
#we are using one example to predict the new email spam can be done or not 0=no spams,1=spams
n_email="click to download image"
k=vectorizer.transform([n_email])#we are convert new text into numerical value
print(model.predict(k))#predicting spam or not
plt.figure(figsize=(10,5))
# Fix: Use numerical indices for x-axis as email strings cannot be plotted directly
plt.scatter(range(len(email)),spams,label="actual data",color="red")
plt.plot(range(len(email)),model.predict(s),label="predicting spam email",color="green")
plt.title("email spam detection")
plt.xlabel("email index") # Changed label to reflect numerical index
plt.ylabel("spams")
plt.show()
