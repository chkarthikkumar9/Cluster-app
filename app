mport importlib
import subprocess
import sys

required_packages = ["pandas", "scikit-learn", "matplotlib", "seaborn"]

for package in required_packages:
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"⚙️ Installing missing package: {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


projects = [
    "Rapido for bikes",
    "Uber for cars",
    "Naukri for jobs"
]

df = pd.DataFrame({'Project_Description': projects})


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Project_Description'])

n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)
df['Cluster'] = kmeans.labels_


score = silhouette_score(X, kmeans.labels_)
print(f"\n✅ Silhouette Score: {score:.2f}\n")


for cluster in range(n_clusters):
    print(f"🔹 Cluster {cluster}:")
    for project in df[df['Cluster'] == cluster]['Project_Description'].values:
        print("   -", project)
    print()


sns.countplot(x=df['Cluster'])
plt.title("Number of Projects per Cluster")
plt.show()
