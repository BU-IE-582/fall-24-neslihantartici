**Data Generation** 

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.decomposition import PCA  
from sklearn.linear\_model import LinearRegression  
from sklearn.preprocessing import StandardScaler  
from sklearn.model\_selection import train\_test\_split  
from sklearn.metrics import mean\_squared\_error, r2\_score

\#Load Data  
input\_data \= pd.read\_csv('hw1\_input.csv')  
real\_data \= pd.read\_csv('hw1\_real.csv')  
image\_data \= pd.read\_csv('hw1\_img.csv')

\#Display the first few rows to verify data  
print("Real Data (S11 Real Part):")  
print(real\_data.head())  
print("\\nInput Data (Geometric Parameters):")  
print(input\_data.head())  
print("\\nImage Data (S11 Imaginary Part):")  
print(image\_data.head())

\#Data Summary and Correlation Matrix  
summary \= input\_data.describe()  
print("\\nData Summary:")  
print(summary)

correlation\_matrix \= input\_data.corr()  
print("\\nCorrelation Matrix:")  
print(correlation\_matrix)

\#Standardize Data for PCA  
scaler \= StandardScaler()  
scaled\_input \= scaler.fit\_transform(input\_data)

\#Applying PCA and Cumulative Explained Variance  
pca \= PCA()  
pca\_components \= pca.fit\_transform(scaled\_input)  
explained\_variance \= pca.explained\_variance\_ratio\_  
cumulative\_variance \= np.cumsum(explained\_variance)

\#Print explained variance for each component and cumulative explained variance  
print("\\nExplained Variance by Each Component:")  
print(explained\_variance)  
print("\\nCumulative Explained Variance:")  
print(cumulative\_variance)

\#Plot Cumulative Explained Variance  
plt.figure(figsize=(10, 6))  
plt.plot(cumulative\_variance, marker='o')  
plt.axhline(0.90, color="r", linestyle="--", label="90% Explained Variance")  
plt.xlabel('Number of Components')  
plt.ylabel('Cumulative Explained Variance')  
plt.title('PCA \- Cumulative Explained Variance by Number of Components')  
plt.legend()  
plt.grid(True)  
plt.show()

\#Determine number of components explaining 90% variance  
pca\_90 \= PCA(n\_components=0.90)  
pca\_90\_components \= pca\_90.fit\_transform(scaled\_input)  
print(f"Number of components to explain 90% variance: {pca\_90.n\_components\_}")

\#S11 Magnitude Calculation and Min Magnitude by Design  
s11\_magnitude \= np.sqrt(real\_data\*\*2 \+ image\_data\*\*2)  
min\_indices \= s11\_magnitude.idxmin(axis=1)  
min\_values \= s11\_magnitude.min(axis=1)

\#Display min S11 magnitude and corresponding frequency index for each design  
result \= pd.DataFrame({  
    'Design': s11\_magnitude.index,  
    'Min Frequency Index': min\_indices,  
    'Min S11 Magnitude': min\_values  
})  
print("\\nMinimum S11 Magnitude by Design:")  
print(result)

\#Plot S11 Magnitude Across Frequency for Selected Designs  
num\_designs\_to\_plot \= 15  
for i in range(num\_designs\_to\_plot):  
    s11\_values \= s11\_magnitude.iloc\[i, :\]  
    plt.plot(s11\_values, label=f"Design {i+1}")  
plt.xlabel("Frequency Index")  
plt.ylabel("S11 Magnitude")  
plt.title("S11 Magnitude Across Frequency Indices for Selected Designs")  
plt.legend(loc="best")  
plt.grid(True)  
plt.show()

\# PCA for Regression Task   
selected\_frequencies \= \[70, 120\]  \# Select specific frequency indices for regression target  
y \= real\_data.iloc\[:, selected\_frequencies\]  \# Real part of S11 at selected frequencies

\#Split data into training and testing sets  
X\_train, X\_test, y\_train, y\_test \= train\_test\_split(pca\_90\_components, y, test\_size=0.2, random\_state=42)

\#Linear Regression Model  
model \= LinearRegression()  
model.fit(X\_train, y\_train)

\#Predictions on test set and model evaluation  
y\_pred \= model.predict(X\_test)  
mse \= mean\_squared\_error(y\_test, y\_pred)  
r2 \= r2\_score(y\_test, y\_pred)

print(f"\\nLinear Regression Model Performance:")  
print(f"Mean Squared Error (MSE): {mse}")  
print(f"R^2 Score: {r2}")

\#Visualizing Predictions vs. True Values  
for i, freq in enumerate(selected\_frequencies):  
    plt.figure(figsize=(8, 5))  
    plt.scatter(y\_test.iloc\[:, i\], y\_pred\[:, i\], alpha=0.7, label=f'Frequency {freq}')  
    plt.plot(\[y\_test.iloc\[:, i\].min(), y\_test.iloc\[:, i\].max()\],  
             \[y\_test.iloc\[:, i\].min(), y\_test.iloc\[:, i\].max()\], 'r--', label="Ideal Fit")  
    plt.xlabel("True S11 (Real Part)")  
    plt.ylabel("Predicted S11 (Real Part)")  
    plt.title(f"S11 Prediction Performance \- Frequency {freq}")  
    plt.legend()  
    plt.grid(True)  
    plt.show()  
