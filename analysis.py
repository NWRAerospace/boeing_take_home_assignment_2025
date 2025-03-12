"""
Take Home Assignment: Data Analysis with Python
I have chosen questions 3, 4, 5
"""
#IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load the data
df_occurrences = pd.read_csv('data/occurrences.csv')
df_aircraft = pd.read_csv('data/aircraft.csv')
df_injuries = pd.read_csv('data/injuries.csv')
df_survivability = pd.read_csv('data/survivability.csv')
df_events_and_phases = pd.read_csv('data/events_and_phases.csv')

"""
Question 3:
3. Describe in detail how you would leverage the Summary field in the Occurrence data to predict future
occurrences. You may consider use of additional libraries in your approach. What would the main
challenges be, and what technical approaches would you take to overcome these? Demonstrate your
approach with relevant data examples.
"""
#load data
df_subset = df_occurrences[['OccID', 'Summary']].head(10)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()  #convert to lowercase
    text = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in text)  #no
    text = ''.join(char for char in text if not char.isdigit())  #no digits
    return text

df_subset["Cleaned_Summary"] = df_subset["Summary"].apply(clean_text)

#get my features from the text
vectorizer = TfidfVectorizer(stop_words="english", max_features=20) #just 20 for showcasing
X = vectorizer.fit_transform(df_subset["Cleaned_Summary"])

tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

#add OccID back in, if I decide to do more with the engineered features
tfidf_df.insert(0, "OccID", df_subset["OccID"].values)

#save features to a little table image
feature_names = vectorizer.get_feature_names_out()
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")
ax.table(cellText=[[name] for name in feature_names],
         colLabels=["TF-IDF Features from first 10 summaries in Occurrences table"],
         cellLoc="center",
         loc="center")

feature_image_path = "outputs/q3_tfidf_features.png"
plt.savefig(feature_image_path, bbox_inches="tight", dpi=300)
plt.close()

"""
Question 4: Develop a model that predicts the probability of surviving a safety incident. What are the key factors
associated with survivability? What are the strengths and weaknesses of your analysis and what would
your next steps be with this model?"
"""
#Step 1: Data Prep
#merge relevant data, clean it
df_aircraft.rename(columns={'occid': 'OccID'}, inplace=True)

#only want survivability of OccID, and I want to change values to be more useful for model
#get rid of unknowns for survivableenum, not useful for predictions
df_surv_simple = df_survivability[['OccID', 'SurvivableEnum']]
df_surv_simple['SurvivableEnum'] = df_surv_simple['SurvivableEnum'].map({2.0: 0, 1.0: 1, 3.0: 2})
df_surv_simple = df_surv_simple.dropna(subset=['SurvivableEnum'])
df_surv_simple = df_surv_simple[df_surv_simple['SurvivableEnum'] != 2]
df_surv_simple = df_surv_simple.reset_index(drop=True)
df_surv_simple = df_surv_simple.groupby('OccID').first().reset_index()
survivability_mapping = {0: 'Non-Survivable', 1: 'Survivable'}

#AIRCRAFT DATA PREPPING
#select only the columns ending in '_DisplayEng', plus OccID
aircraft_display_cols = [col for col in df_aircraft.columns if col.endswith('_DisplayEng')]
aircraft_display_cols = ['OccID'] + aircraft_display_cols
aircraft_simplified = df_aircraft[aircraft_display_cols].groupby('OccID').first().reset_index()
#convert the displayend cols to strings
for col in aircraft_display_cols[1:]:
    aircraft_simplified[col] = aircraft_simplified[col].astype(str)


#OCCURRENCES DATA PREPPING
df_occurrences['AirportID_AirportName'] = df_occurrences['AirportID_AirportName'].astype(str)
df_occurrences['ICAO_DisplayEng'] = df_occurrences['ICAO_DisplayEng'].astype(str)
occ_simple = df_occurrences[['OccID', 'AirportID_AirportName', 'ICAO_DisplayEng']]
occ_simple = occ_simple.groupby('OccID').first().reset_index()

#EVENTS DATA
#convert events to strings
df_events_and_phases['EventID_DisplayEng'] = df_events_and_phases['EventID_DisplayEng'].astype(str)
event_counts = df_events_and_phases['EventID_DisplayEng'].value_counts()
#ignore events that dont occur often
frequent_events = event_counts[event_counts >= 10].index
#one-hot encoded df of events
df_filtered = df_events_and_phases[df_events_and_phases['EventID_DisplayEng'].isin(frequent_events)]
one_hot_events = pd.crosstab(df_filtered['OccID'], df_filtered['EventID_DisplayEng'])
one_hot_events = one_hot_events.groupby('OccID').first().reset_index()

#merging all the cleaned data into one df for modelling
surv_air_df = df_surv_simple.merge(aircraft_simplified, on='OccID', how='left')
surv_air_occ_df = surv_air_df.merge(occ_simple, on='OccID', how='left')
merged_data = surv_air_occ_df.merge(one_hot_events, on='OccID', how='left')
merged_data.drop(columns=['nan', 'other'], inplace=True)

#separate target and features
X = merged_data.drop(['SurvivableEnum', 'OccID'], axis=1)
y = merged_data['SurvivableEnum']

#categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_transformer = SimpleImputer(strategy='constant', fill_value=0)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

#randomforest modelling
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=300, random_state=42))
])
#42 is always the best random state
#this is sparta

#splitting into train and test sets at 30% for testing, stratifying on y for imbalanced data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)

#fit clf model
clf = clf.fit(X_train, y_train)

#make y pred
y_pred = clf.predict(X_test)

#extract the feature names from clf
feature_names = clf.named_steps['preprocessor'].get_feature_names_out()
feature_importances = pd.Series(clf.named_steps['classifier'].feature_importances_, index=feature_names)

#eval
#predictions on the test set
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

#class report
class_report = classification_report(y_test, y_pred)
with open('outputs/q4_survivability_classification_report.txt', 'w') as f:
    f.write("Question 4 Classification Report for Survivability Prediction Model (RandomForest):\n\n")
    f.write(class_report)
    f.write(f"ROC AUC Score: {roc_auc:.4f}")

#feature importances
feature_names = clf.named_steps['preprocessor'].get_feature_names_out()
feature_importances = pd.Series(clf.named_steps['classifier'].feature_importances_, index=feature_names)

#store feature importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

#importance
importance_df = importance_df.sort_values('Importance', ascending=False)

#plot important features
plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'].head(10), importance_df['Importance'].head(10))
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Features for Predicting Incident Survivability')
plt.tight_layout()
plt.savefig('outputs/q4_survivability_feature_importance.png')
plt.close()

#get roccurve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

#plot roc curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Incident Survivability Prediction')
plt.legend()
plt.grid()

#save roc to outptus
plt.savefig('outputs/q4_survivability_roc_curve.png', dpi=300)
plt.close()

"""
Question 5: What ICAO event categories are most common at Canadian airports? Is there any trend or pattern evident
in these Canadian events?
"""
import pandas as pd
import matplotlib.pyplot as plt

df_occ = pd.read_csv('data/occurrences.csv', parse_dates=['OccDate'])

#only canadian icao event
df_canada = df_occ[df_occ['AirportID_CountryID_DisplayEng'] == 'CANADA']

#number of each icao event
icao_counts = df_canada['ICAO_DisplayEng'].value_counts().head(10)

#make a hori bar plot
plt.figure(figsize=(10,6))
icao_counts.plot(kind='barh', color='skyblue')
plt.xlabel('Number of Incidents')
plt.ylabel('ICAO Events')
plt.title('ICAO Events at CDN Airports')
plt.yticks(fontsize=6)  # Adjust the fontsize as needed
plt.tight_layout()
plt.savefig('outputs/q5_icao_event_categories.png')

#extracting year and icao top 5 categories
df_canada['Year'] = df_canada['OccDate'].dt.year
top_5_categories = df_canada['ICAO_DisplayEng'].value_counts().nlargest(5).index

#make a top 5 of yearly icao events
icao_trends = df_canada.groupby(['OccDate', 'ICAO_DisplayEng']).size().reset_index(name='Count')
icao_trends['Year'] = pd.to_datetime(icao_trends['OccDate']).dt.year
icao_yearly = icao_trends.groupby(['Year', 'ICAO_DisplayEng'])['Count'].sum().unstack(fill_value=0)
icao_yearly_recent = icao_yearly[icao_yearly.index >= 2014]
icao_yearly_top5 = icao_yearly_recent[top_5_categories]

#plot the top icao events
icao_yearly_top5.plot(kind='line', figsize=(14,8))
plt.title('ICAO Event Categories Trends (Canada)')
plt.ylabel('Number of Incidents')
plt.xlabel('Year')
plt.tight_layout()
plt.legend(title='ICAO Event Category', loc='upper left')
plt.savefig('outputs/q5_icao_event_trends.png')

