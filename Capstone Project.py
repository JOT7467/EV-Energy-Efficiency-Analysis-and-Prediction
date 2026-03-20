###################################################################################
#Determinant and Prediction of EV Energy Efficiency Prediction: Canada(2012 - 2026)
####################################################################################

###################################################################################
#1.Data Processing
##################################################################################

####################################----Installing Libraries------##################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

###################################----Loading Dataset----------##############################################
EV_df = pd.read_csv('Input/EV dataset.csv')

#######################################---- Data Inspection----##########################################


##############################################################
#1. Remove hidden spaces from all column names in the dataset
################################################################
EV_df.columns = EV_df.columns.str.strip()

########################################################
#2. Review column names, datatype and Non-null counts
########################################################
print(EV_df.columns)
print(EV_df.head())
print(EV_df.info())

################################################################################################
#3. removing 3 columns from the dataset
#Explanation: I excluded these 3 variables because:
#Three variables — CO2 emissions (g/km), CO2 rating, and Smog rating — were removed during preprocessing
# because all records contained either zero or missing values. Variables without variation do not provide
# meaningful information for statistical analysis or machine learning models, and retaining them could introduce unnecessary noise into the dataset.
####################################################################################################
EV_df = EV_df.drop(columns=[
    "CO2 emissions (g/km)",
    "CO2 rating",
    "Smog rating"
])


##################################################################
#4. convert all numeric columns in the EV dataset to numeric
#Explanation
#Convert all text in the numeric column to numeric
#errors = "Coerce" converts invalid values to NaN
##################################################################

numeric_cols = [
"Motor (kW)",
"City (kWh/100 km)",
"Highway (kWh/100 km)",
"Combined (kWh/100 km)",
"Range (km)",
"Recharge time (h)"
]

for col in numeric_cols:
    EV_df[col] = pd.to_numeric(EV_df[col], errors="coerce")



#################################################################################
#5. Removing Missing value from the dataset
#Explanation: Energy efficiency requires the combined consumption variable, so rows missing it must be removed
###################################################################################

EV_df = EV_df.dropna(subset=["Combined (kWh/100 km)"])

#######################################################################
#6.Clean Categorical Variables
#Explanation: Removes extra spaces that can create duplicate categories
########################################################################
EV_df["Make"] = EV_df["Make"].str.strip()
EV_df["Vehicle class"] = EV_df["Vehicle class"].str.strip()
EV_df["Model"] = EV_df["Model"].str.strip()

##################################################################################
#Energy efficiency measures distance traveled per unit of electricity
#This converts energy consumption into efficiency
#for example if a vehicle consumes 18 kwh per 100 km
#the efficiency of the vehicle becomes 10/18 = 5.56km/kwh
#higher value means more efficiency
#####################################################################################
EV_df["Energy Efficiency (km/kWh)"] = 100 / EV_df["Combined (kWh/100 km)"]
EV_df["City Efficiency (km/kWh)"] = 100 / EV_df["City (kWh/100 km)"]
EV_df["Highway Efficiency (km/kWh)"] = 100 / EV_df["Highway (kWh/100 km)"]

#####################################################
#8. Remove all impossible values
#####################################################
EV_df = EV_df[EV_df["Energy Efficiency (km/kWh)"] > 0]
EV_df = EV_df[EV_df["City Efficiency (km/kWh)"] > 0]
EV_df = EV_df[EV_df["Highway Efficiency (km/kWh)"] > 0]

#########################################################
#9.Power to efficiency ratio
#########################################################
EV_df["power_efficiency_ratio"] = EV_df["Motor (kW)"] / EV_df["Energy Efficiency (km/kWh)"]
EV_df["City_power_efficiency_ratio"] = EV_df["Motor (kW)"] / EV_df["City Efficiency (km/kWh)"]
EV_df["Highway_power_efficiency_ratio"] = EV_df["Motor (kW)"] / EV_df["Highway Efficiency (km/kWh)"]

#####################################################################
#10. Charging Speed; km gain per hour of charging
#########################################################################
EV_df["charging_speed"] = EV_df["Range (km)"] / EV_df["Recharge time (h)"]

#Remove any impossible value
EV_df = EV_df[EV_df["charging_speed"] > 0]


#####################################
#Save the cleaned Data
#####################################
EV_df.to_csv("cleaned_EV_dataset.csv", index=False)


##########################---------Explorative Data Analysis------##################################

EV_df = pd.read_csv("cleaned_EV_dataset.csv")

#1. Dataset Overview: Returns the number of rows and columns in the dataset.
print(EV_df.shape)
print(EV_df.head())  #Displays the first few observations to confirm the dataset structure and variables.

################################################################################
#2. Summary Statistics: compute summary statistics for all numeric variables
#such as the mean, max, min, standard Deviation and quartiles
#This helps identify ranges and variability in variables
#################################################################################
print(EV_df.describe())


#############--------#3. Distribution of Variables-------#######################

#A. Motor Vehicle Distribution : Shows how powerful most EV motors are.
#High-power EVs often trade efficiency for performance.

sns.histplot(EV_df["Motor (kW)"], bins=30)
plt.title("Distribution of EV Motor Power")
plt.xlabel("Motor Power (kW)")
plt.show()

#B. Recharge Distribution : Identifies typical charging durations across EV models

sns.histplot(EV_df["Recharge time (h)"], bins=30)
plt.title("Distribution of Recharge Time")
plt.xlabel("Recharge Time (hours)")
plt.show()

#C. Range Distribution: Shows how far EVs can travel on a full charge

sns.histplot(EV_df["Range (km)"], bins=30)
plt.title("Distribution of EV Range")
plt.xlabel("Range (km)")
plt.show()

#D. Efficiency Distribution: Shows how efficient most EVs are

sns.histplot(EV_df["Energy Efficiency (km/kWh)"], bins=30)
plt.title("Distribution of EV Energy Efficiency")
plt.show()

########------ Categorical Features Analysis---------################################
#E. Top Manufacturer: Identifies which manufacturers dominate the EV dataset

top_makes = EV_df["Make"].value_counts().head(10)
print(top_makes)

sns.barplot(x=top_makes.index, y=top_makes.values)
plt.xticks(rotation=45)
plt.title("Top EV Manufacturers")
plt.show()

#F. E Vehicle class Distribution

top_class = EV_df["Vehicle class"].value_counts()
print(top_class)

sns.barplot(x=top_class.index, y=top_class.values)
plt.xticks(rotation=50)
plt.title("EV Class Distribution")
plt.show()


#######################------Relationship Analysis-------------##########################

#G. Motor KW vs Efficiency: Tests whether high-performance EVs sacrifice efficiency(often negative relationship)

sns.scatterplot(
    x="Motor (kW)",
    y="Energy Efficiency (km/kWh)",
    data=EV_df
)

plt.title("Motor Power vs Energy Efficiency")
plt.show()

#H.Model Year vs Efficiency
#Shows whether EV efficiency has improved over time due to technology
sns.scatterplot(
    x="Model year",
    y="Energy Efficiency (km/kWh)",
    data=EV_df
)

plt.title("Model Year vs Energy Efficiency")
plt.show()


#J.Recharge Time vs Efficiency: Tests whether vehicles with longer recharge times differ in efficiency
sns.scatterplot(
    x="Recharge time (h)",
    y="Energy Efficiency (km/kWh)",
    data=EV_df
)

plt.title("Recharge Time vs Efficiency")
plt.show()


#K. Range vs Efficiency
#Long-range EVs require larger batteries and therefore use more energy
#Efficient EVs tend to have moderate ranges

sns.scatterplot(
    x="Range (km)",
    y="Energy Efficiency (km/kWh)",
    data=EV_df
)

plt.title("EV Range vs Energy Efficiency")
plt.xlabel("Range (km)")
plt.ylabel("Energy Efficiency (km/kWh)")
plt.show()

#K.Correlation Matrix
#Identifies which variables correlate most strongly with energy efficiency

numeric = EV_df.select_dtypes(include=np.number)

sns.heatmap(numeric.corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()


#l. Tesla vs other manufacturers test
EV_df["Tesla"] = EV_df["Make"].apply(lambda x: 1 if x == "Tesla" else 0)

sns.boxplot(
    x="Tesla",
    y="Energy Efficiency (km/kWh)",
    data=EV_df
)

plt.xticks([0,1], ["Other Manufacturers", "Tesla"])
plt.title("Tesla vs Other EV Efficiency")
plt.show()


#Statistical Test
#This test evaluates whether the average efficiency difference is statistically significan
#Interpretation: There is a statistically significant difference in energy efficiency between Tesla vehicles and other manufacturers, t(1202) = 11.83, p = 1.29×10⁻³⁰
#
from scipy.stats import ttest_ind

tesla_eff = EV_df[EV_df["Tesla"] == 1]["Energy Efficiency (km/kWh)"]
other_eff = EV_df[EV_df["Tesla"] == 0]["Energy Efficiency (km/kWh)"]

print(ttest_ind(tesla_eff, other_eff))


#Model Year VS Efficiency
sns.regplot(
    x="Model year",
    y="Energy Efficiency (km/kWh)",
    data=EV_df
)

plt.title("EV Energy Efficiency Improvement Over Time")
plt.xlabel("Model Year")
plt.ylabel("Energy Efficiency (km/kWh)")
plt.show()

##########################-------------EV Technology Clusters------###############################
#1. Cluster 0 → High-efficiency vehicles
#2. Cluster 1 → Balanced performance EVs
#3. Cluster 2 → High-performance EVs


# Selecting Variables for clustering

from sklearn.preprocessing import StandardScaler

features = EV_df[
    ["Motor (kW)", "Range (km)", "Energy Efficiency (km/kWh)", "Recharge time (h)"]
]

#Scaling Data: Scaling ensures variables with different units are comparable
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

#K-Mean clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
EV_df["Cluster"] = kmeans.fit_predict(scaled_features)

#Visualize Clusters
sns.scatterplot(
    x="Motor (kW)",
    y="Energy Efficiency (km/kWh)",
    hue="Cluster",
    data=EV_df,
    palette="viridis"
)

plt.title("EV Technology Clusters")
plt.show()


#########################------------Machine Learning Development--------##########################

#Goal : Predict Energy Efficiency (km/kWh) using vehicle characteristics

################################################################################
#1. Importing Libraries
#These libraries will help me:
#• Split data into training and testing sets
#• Build machine learning models
#• Evaluate prediction performance
################################################################################
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

#################################################################
#2.Featured Selection
#these are the list of relevant predictors for the model
#Motor power → affects energy consumption
#Recharge time → relates to battery capacity
#Range → reflects battery size
#Model year → captures technological improvements
################################################################
model_features = [
    "Motor (kW)",
    "Recharge time (h)",
    "Range (km)",
    "Model year"
]

X = EV_df[model_features]

y = EV_df["Energy Efficiency (km/kWh)"]

#################################################################################
#3. Train Test Split
#I split the dataset into
# 80% as training dataset
# 20% as testing data
# The model learns on training data and is evaluated on unseen testing data
#################################################################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

########################################################################################
#4. Model 1 - Linear Regression
# Linear regression assumes a linear relationship between features and efficiency
##########################################################################################

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

#print(lr_predictions)

#############################################################################
# Evaluating Linear Regression Model Performance
# RMSE -prediction error magnitude
# R² - proportion of variance explained
# Higher R² = better model performance
# output: Linear Regression RMSE: 0.6066
# Linear Regression R²: 0.5230
############################################################################
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
lr_r2 = r2_score(y_test, lr_predictions)

print("Linear Regression RMSE:", lr_rmse)
print("Linear Regression R²:", lr_r2)


###################################################################################################
# Model 2 - Random Forest Model
#Random Forest is an ensemble machine learning algorithm that combines many decision trees.
#Advantages:
#• captures nonlinear relationships
#• handles complex interactions
#• usually more accurate than linear models
###################################################################################################
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

################################################################
# Evaluating Random Forest Model
# Result : Random Forest RMSE: 0.3045
# Random Forest R²: 0.8798
################################################################
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

print("Random Forest RMSE:", rf_rmse)
print("Random Forest R²:", rf_r2)


##############################################################################
#Model Comparison
# Best Model Means: Lower RMSE and high R²
#               Model      RMSE        R2
#m1- Linear Regression  0.606686  0.523076
#m2-     Random Forest  0.304514  0.879847
##############################################################################

model_results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "RMSE": [lr_rmse, rf_rmse],
    "R2": [lr_r2, rf_r2]
})

print(model_results)


#############################################################################
#Features Importance Analysis
#This analysis identify which variable most influence Efficiency
############################################################################
import matplotlib.pyplot as plt

importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": model_features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print(feature_importance)

###############################################
#Visualization
##############################################

plt.barh(feature_importance["Feature"], feature_importance["Importance"])

plt.title("Feature Importance for EV Efficiency Prediction")
plt.xlabel("Importance Score")

plt.show()


###########################################################################
#Prediction Visualization
#If predictions are good, points will lie close to the 45° line
##########################################################################
plt.scatter(y_test, rf_predictions)

plt.xlabel("Actual Efficiency")
plt.ylabel("Predicted Efficiency")

plt.title("Actual vs Predicted EV Efficiency")

plt.show()


###################--------Generative AI EV Analyst---------------####################
import os
from openai import OpenAI

#############################################################################
#Create Summary Statistics
##############################################################################
summary = EV_df.describe().to_string()

print(summary)


ai_summary = """
Using generative AI assistance, insights were generated to interpret
the EV efficiency dataset. Key findings include:

- EV efficiency improves steadily across model years
- Motor power strongly influences energy efficiency
- Higher performance vehicles tend to consume more energy
- Battery technology improvements have increased vehicle range
"""

print(ai_summary)




##############################################################################
#Connect to OpenAI API KEY
##############################################################################
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

############################################################################
#Asking AI to Generate EV Insight
###########################################################################
prompt = f"""
You are an EV energy analyst.

Analyze the following dataset summary and provide key insights
about electric vehicle efficiency trends.

Dataset Summary:
{summary}

Focus on:
- efficiency trends
- vehicle performance tradeoffs
- technological improvements
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role":"user","content":prompt}
    ]
)

print(response.choices[0].message.content)

############################################################################
# AI Explanation of Machine Learning Results
###########################################################################
model_results = """
Linear Regression R2: 0.52
Random Forest R2: 0.88

Feature Importance:
Motor Power: 0.46
Range: 0.28
Recharge Time: 0.18
Model Year: 0.08
"""

prompt = f"""
Explain the following EV machine learning results
in simple terms for a technical report.

{model_results}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role":"user","content":prompt}]
)

print(response.choices[0].message.content)