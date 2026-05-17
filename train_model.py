import pyspark
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# create spark session
spark = pyspark.sql.SparkSession.builder \
    .appName("train_model") \
    .master("local[*]") \
    .getOrCreate()

# load gold tables
profile = spark.read.parquet("datamart/gold/risk_profile/*")
behaviour = spark.read.parquet("datamart/gold/risk_behaviour/*")
performance = spark.read.parquet("datamart/gold/risk_performance/*")

# join datasets
df = profile.join(
    behaviour,
    on=["Customer_ID", "snapshot_date"],
    how="left"
)

df = df.join(
    performance,
    on=["Customer_ID", "snapshot_date"],
    how="inner"
)

print("profile count:", profile.count())
print("behaviour count:", behaviour.count())
print("performance count:", performance.count())

print("profile columns:", profile.columns)
print("behaviour columns:", behaviour.columns)
print("performance columns:", performance.columns)

df1 = profile.join(
    behaviour,
    on=["Customer_ID", "snapshot_date"],
    how="left"
)

print("after profile + behaviour join:", df1.count())

df = df1.join(
    performance.select("Customer_ID", "label", "label_def"),
    on="Customer_ID",
    how="inner"
)

print("after adding performance label:", df.count())

df.select("Customer_ID", "label").show(10, False)

# convert to pandas
pdf = df.toPandas()

# remove non-feature columns
drop_cols = [
    "Customer_ID",
    "snapshot_date",
    "loan_id",
    "label_def"
]

X = pdf.drop(columns=["label"] + drop_cols, errors="ignore")
y = pdf["label"]

# convert list columns to string
for c in X.columns:
    X[c] = X[c].astype(str)

# encode categoricals
X = pd.get_dummies(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# train model
model = RandomForestClassifier()

model.fit(X_train, y_train)

# predictions
preds = model.predict(X_test)

# outputs
print("Dataset shape:", X.shape)

print("Label distribution:")
print(y.value_counts())

print("Model trained successfully")

print(classification_report(y_test, preds))