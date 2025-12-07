import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# set the style to whitegrid
sns.set_theme(style="whitegrid")

# Load dataset
df = pd.read_csv("Motor_Vehicle_Collisions_-_Crashes.csv", low_memory=False)

# ---------------------------------------
# EDA
# ---------------------------------------

# display top 5 rows of the dataframe
print(df.head())

# display the overall structure of the dataframe
df.info(show_counts=True)

# display the count of rows and columns
print("Shape:", df.shape)

# save mean, max etc of all numeric columns in a csv file
summary = df.describe()
summary.to_csv("summary_statistics.csv", index=True)

# Print whether a value is present or not for a column
print(df.isna())
# Check for missing values per column
print(df.isna().sum())

# Check for duplicate rows  
print("Duplicate rows: ", df.duplicated())
print("Sum of duplicate Rows: ", df.duplicated().sum())

print(df.columns.tolist())

# Example: check duplicates based on date, time, and street info
duplicates = df.duplicated(subset=['CRASH DATE', 'CRASH TIME', 'ON STREET NAME', 'CROSS STREET NAME'])
print("Number of duplicate crashes:", duplicates.sum())

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# dup_rows = df[df.duplicated(subset=['CRASH DATE', 'CRASH TIME', 'ON STREET NAME', 'CROSS STREET NAME'], keep=False)]
# print(dup_rows)

#---------------------------------------
# Data Wrangling
#---------------------------------------

# Handle missing values
df['BOROUGH'] = df['BOROUGH'].fillna("UNKNOWN")
df['ZIP CODE'] = df['ZIP CODE'].fillna("UNKNOWN")

df['CONTRIBUTING FACTOR VEHICLE 1'] = df['CONTRIBUTING FACTOR VEHICLE 1'].fillna("Unspecified")

# Remove duplicates
df = df.drop_duplicates()

# crash date and time converted to date time and time objects respectively
df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'], errors='coerce')
df['CRASH TIME'] = pd.to_datetime(df['CRASH TIME'], format="%H:%M", errors='coerce')

# Extract useful time features
df['YEAR'] = df['CRASH DATE'].dt.year
df['MONTH'] = df['CRASH DATE'].dt.month
df['DAY_OF_WEEK'] = df['CRASH DATE'].dt.day_name()
df['HOUR'] = df['CRASH TIME'].dt.hour

# Convert columns to numeric types
num_cols = [
    'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED',
    'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED',
    'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED',
    'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED',
    'LATITUDE', 'LONGITUDE'
]

df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

#---------------------------------------
# DATA VISUALIZATIONS 
#---------------------------------------

# Top 10 dangerous streets
top_streets = df['ON STREET NAME'].value_counts().head(10)

plt.figure(figsize=(12,6))
sns.barplot(x=top_streets.values, y=top_streets.index, palette='Reds_r')
plt.title("Top 10 Streets with Most Collisions")
plt.xlabel("Number of Collisions")
plt.ylabel("Street Name")
plt.tight_layout()
plt.show()

inj_cols = [
    'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED',
    'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED',
    'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED',
    'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED',
]

# visualization of injured and killed people
df[inj_cols].hist(figsize=(14,10), bins=30)
plt.suptitle("Distribution of Injury and Fatality Counts")
plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)  # Add padding
plt.show()

# Heatmap of injuries vs deaths
plt.figure(figsize=(8,6))
sns.heatmap(df[inj_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Different Injury/Death Types")
plt.tight_layout()
plt.show()

# crash counts by borough
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='BOROUGH', order=df['BOROUGH'].value_counts().index)
plt.title("Total Crashes by Borough")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# top 10 vehicle types involved in crashes
plt.figure(figsize=(12,6))
sns.countplot(
    data=df, 
    x='VEHICLE TYPE CODE 1',
    order=df['VEHICLE TYPE CODE 1'].value_counts().index[:10]
)
plt.title("Top 10 Vehicle Types Involved in Crashes")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# correlation heatmap
plt.figure(figsize=(10,7))
sns.heatmap(df[inj_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Injury/Fatality Variables")
plt.tight_layout()
plt.show()

# collisions over time
df_time = df.set_index('CRASH DATE')

monthly_counts = df_time.resample('M').size()

plt.figure(figsize=(14,6))
monthly_counts.plot()
plt.title("Monthly Collision Trend")
plt.ylabel("Number of Crashes")
plt.show()

# crashes peak during rush hours
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='HOUR', color='steelblue')
plt.title("Crashes by Hour of Day")
plt.show()

# crashes on weekends vs weekdays
plt.figure(figsize=(12,6))
sns.countplot(
    data=df,
    x='DAY_OF_WEEK',
    order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
)
plt.title("Crashes by Day of Week")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Drop rows with missing coordinates
df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

plt.figure(figsize=(10, 10))

# Scatter plot
plt.scatter(
    df["LONGITUDE"], 
    df["LATITUDE"], 
    s=5,                  # small dot size
    alpha=0.4,            # transparency
    edgecolors='none'
)

plt.title("Scatter Map of Collision Locations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.grid(True)
plt.show()

#---------------------------------------
# MACHINE LEARNING MODELS
#---------------------------------------

# Injury prediction

# Create target variable
df["ANY_INJURY"] = (df["NUMBER OF PERSONS INJURED"] > 0).astype(int)

# Extract hour FAST
df["HOUR"] = df["CRASH TIME"].str.slice(0, 2)

# Convert hour to numeric
df["HOUR"] = pd.to_numeric(df["HOUR"], errors="coerce")

# Keep only valid rows
df = df.dropna(subset=["LATITUDE", "LONGITUDE", "HOUR"])

# Reduce dataset to prevent freezing
df = df.sample(200000, random_state=42)

# Features
X = df[["LATITUDE", "LONGITUDE", "HOUR"]]
y = df["ANY_INJURY"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Faster Random Forest
model = RandomForestClassifier(
    n_estimators=70,
    max_depth=12,
    n_jobs=-1,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Results
print(classification_report(y_test, pred))
