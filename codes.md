\#Libraries  
import pandas as pd  
import numpy as np  
import zipfile  
import matplotlib.pyplot as plt  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.model\_selection import train\_test\_split  
from sklearn.metrics import classification\_report, confusion\_matrix, ConfusionMatrixDisplay  
import joblib

\#Load and Extract ZIP File  
zip\_path \= 'match\_data.zip'  
with zipfile.ZipFile(zip\_path, 'r') as z:  
    file\_names \= z.namelist()  
    print("Files Inside the ZIP:", file\_names)  
    with z.open(file\_names\[0\]) as f:  
        data \= pd.read\_csv(f)

\#Data Cleaning: Remove Suspended or Stopped Rows  
clean\_data \= data\[(data\['suspended'\] \== False) & (data\['stopped'\] \== False)\].copy()  
clean\_data.reset\_index(drop=True, inplace=True)

\#Handle Missing Values  
required\_columns \= \['1', 'X', '2'\]  
clean\_data.dropna(subset=required\_columns, inplace=True)

\#Convert columns to numeric types  
for col in required\_columns:  
    clean\_data\[col\] \= pd.to\_numeric(clean\_data\[col\], errors='coerce')  
clean\_data.dropna(subset=required\_columns, inplace=True)

\#Calculate Probabilities  
clean\_data\['P\_home'\] \= 1 / clean\_data\['1'\]  
clean\_data\['P\_away'\] \= 1 / clean\_data\['2'\]  
clean\_data\['P\_draw'\] \= 1 / clean\_data\['X'\]

\#Normalize Probabilities  
total\_prob \= clean\_data\[\['P\_home', 'P\_draw', 'P\_away'\]\].sum(axis=1)  
clean\_data\['P\_home\_norm'\] \= clean\_data\['P\_home'\] / total\_prob  
clean\_data\['P\_away\_norm'\] \= clean\_data\['P\_away'\] / total\_prob  
clean\_data\['P\_draw\_norm'\] \= clean\_data\['P\_draw'\] / total\_prob

\#Calculate Home-Away Difference  
clean\_data\['home\_away\_diff'\] \= clean\_data\['P\_home\_norm'\] \- clean\_data\['P\_away\_norm'\]

\#Noise Identification: Late Goals and Early Red Cards  
if 'Yellowred Cards \- home' in clean\_data.columns and 'Yellowred Cards \- away' in clean\_data.columns:  
    late\_goals \= clean\_data\[(clean\_data\['minute'\] \>= 90\) & (clean\_data\['current\_state'\] \== '1')\]  
    early\_red\_cards \= clean\_data\[  
        (clean\_data\['minute'\] \<= 15\) &   
        ((clean\_data\['Yellowred Cards \- home'\] \> 0\) | (clean\_data\['Yellowred Cards \- away'\] \> 0))  
    \]  
else:  
    print("Red card data is not available. Skipping this analysis.")

\#Remove Noisy Matches  
noise\_removed\_data \= clean\_data.copy()  
if not late\_goals.empty:  
    noise\_removed\_data \= noise\_removed\_data\[\~noise\_removed\_data\['fixture\_id'\].isin(late\_goals\['fixture\_id'\])\]  
if not early\_red\_cards.empty:  
    noise\_removed\_data \= noise\_removed\_data\[\~noise\_removed\_data\['fixture\_id'\].isin(early\_red\_cards\['fixture\_id'\])\]

\#Bin Home-Away Differences and Calculate Draw Probabilities  
bins \= \[-1, \-0.8, \-0.6, \-0.4, \-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1\]  
noise\_removed\_data\['home\_away\_diff\_bin'\] \= pd.cut(noise\_removed\_data\['home\_away\_diff'\], bins)

\#Compute Observed Draw Probabilities  
bin\_counts \= noise\_removed\_data\['home\_away\_diff\_bin'\].value\_counts().sort\_index()  
bin\_draws \= noise\_removed\_data\[noise\_removed\_data\['result'\] \== 'X'\]\['home\_away\_diff\_bin'\].value\_counts().sort\_index()  
observed\_draw\_prob \= (bin\_draws / bin\_counts).fillna(0)

\#Display Results  
print("Draw Probabilities After Noise Removal:\\n", observed\_draw\_prob)

\#Comparison Plot: Before and After Noise Removal  
plt.figure(figsize=(12, 6))

\#Before Noise Removal  
plt.subplot(1, 2, 1\)  
plt.scatter(clean\_data\['home\_away\_diff'\], clean\_data\['P\_draw\_norm'\], alpha=0.5, color='blue', label='Original')  
plt.xlabel('P(Home Win) \- P(Away Win)')  
plt.ylabel('P(Draw)')  
plt.title('Before Noise Removal')  
plt.grid()

\#After Noise Removal  
plt.subplot(1, 2, 2\)  
plt.scatter(noise\_removed\_data\['home\_away\_diff'\], noise\_removed\_data\['P\_draw\_norm'\], alpha=0.5, color='green', label='Noise Removed')  
plt.xlabel('P(Home Win) \- P(Away Win)')  
plt.ylabel('P(Draw)')  
plt.title('After Noise Removal')  
plt.grid()

\#Show Comparison Plot  
plt.tight\_layout()  
plt.show()

\#Decision Tree Model  
features \= noise\_removed\_data\[\['P\_home\_norm', 'P\_draw\_norm', 'P\_away\_norm'\]\]  
target \= noise\_removed\_data\['result'\]

\#Split into Training and Testing Sets  
X\_train, X\_test, y\_train, y\_test \= train\_test\_split(features, target, test\_size=0.3, random\_state=42)

\#Train the Decision Tree Classifier  
model \= DecisionTreeClassifier(max\_depth=5, random\_state=42)  
model.fit(X\_train, y\_train)

\#Evaluate Model Performance  
y\_pred \= model.predict(X\_test)  
print("Decision Tree Model Performance:")  
print(classification\_report(y\_test, y\_pred))

\#Confusion Matrix  
cm \= confusion\_matrix(y\_test, y\_pred, labels=model.classes\_)  
disp \= ConfusionMatrixDisplay(confusion\_matrix=cm, display\_labels=model.classes\_)  
disp.plot(cmap='Blues')  
plt.title("Confusion Matrix \- Decision Tree")  
plt.show()

\#Feature Importances  
importances \= model.feature\_importances\_  
print("Feature Importances:", dict(zip(features.columns, importances)))  
