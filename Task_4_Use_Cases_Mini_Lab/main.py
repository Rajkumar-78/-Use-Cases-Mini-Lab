
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(texts):
    return model.encode(texts, convert_to_numpy=True)

def find_similar(query, items, embeddings, top_n=3):
    query_embedding = create_embeddings([query])
    scores = cosine_similarity(query_embedding, embeddings)[0]
    indices = np.argsort(scores)[::-1][:top_n]

    for index in indices:
        print(f"{scores[index]:.3f} | {items[index]}")

print("\nTASK 4: EMBEDDING USE CASES MINI LAB")

faq = [
    "How can I reset my password?",
    "How do I change my account password?",
    "How can I update my email address?",
    "How do I contact customer support?",
    "What are the office working hours?",
    "How many casual leaves are available?",
    "How can I apply for annual leave?",
    "How do I check my salary details?",
    "How can I download my payslip?",
    "What should I do if my account is locked?",
    "How do I update my phone number?",
    "How can I request a refund?",
    "What payment methods are accepted?",
    "How do I track my order?",
    "How can I cancel my subscription?"
]

faq_embeddings = create_embeddings(faq)

print("\n1. SEMANTIC SEARCH")

queries = [
    "I cannot remember my login password",
    "I want to know my holiday allowance",
    "Where can I find my salary slip?"
]

for query in queries:
    print("\nQuery:", query)
    find_similar(query, faq, faq_embeddings, 3)

complaints = [
    "My internet connection is very slow.",
    "The network speed is poor.",
    "I cannot connect to the internet.",
    "My laptop battery drains quickly.",
    "The battery does not last long.",
    "My computer battery is not charging.",
    "The application crashes when I open it.",
    "The app closes unexpectedly.",
    "The software stops working after startup.",
    "My payment was deducted twice.",
    "I was charged two times for one order.",
    "My refund has not arrived."
]

complaint_embeddings = create_embeddings(complaints)
complaint_matrix = cosine_similarity(complaint_embeddings)

print("\n2. DUPLICATE COMPLAINT DETECTION")

threshold = 0.65
duplicate_count = 0

for i in range(len(complaints)):
    for j in range(i + 1, len(complaints)):
        score = complaint_matrix[i][j]

        if score >= threshold:
            print(f"\nSimilarity: {score:.3f}")
            print("Complaint 1:", complaints[i])
            print("Complaint 2:", complaints[j])
            duplicate_count += 1

print("\nPotential duplicate pairs:", duplicate_count)

headlines = [
    "Government announces new education policy",
    "Schools introduce digital learning programs",
    "Universities expand online courses",
    "Students prepare for annual examinations",
    "New technology improves classroom learning",
    "Stock markets rise after positive earnings",
    "Investors react to company profit reports",
    "Bank announces changes in interest rates",
    "Global markets face economic uncertainty",
    "Technology companies report strong growth",
    "New smartphone launches with advanced camera",
    "Artificial intelligence changes software development",
    "Scientists develop a faster computer chip",
    "Cybersecurity becomes a major technology concern",
    "New electric vehicle technology is introduced",
    "Cricket team wins international tournament",
    "Football club announces new head coach",
    "Athletes prepare for upcoming championship",
    "Tennis player reaches tournament final",
    "Sports federation announces new competition",
    "Heavy rainfall affects several Indian cities",
    "Weather department issues rain warning",
    "Temperature rises across southern India",
    "Cyclone warning issued for coastal areas",
    "Scientists study climate change effects",
    "Hospital introduces new patient care system",
    "Researchers develop a new medical treatment",
    "Doctors discuss advances in cancer research",
    "Public health campaign promotes vaccination",
    "New medical technology improves diagnosis"
]

headline_embeddings = create_embeddings(headlines)

print("\n3. NEWS HEADLINE CLUSTERING")

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(headline_embeddings)

cluster_names = {
    0: "Education and Learning",
    1: "Business and Finance",
    2: "Technology and Innovation",
    3: "Sports",
    4: "Weather and Healthcare"
}

for cluster_id in range(5):
    print("\nCluster:", cluster_names[cluster_id])

    for i, headline in enumerate(headlines):
        if cluster_labels[i] == cluster_id:
            print("-", headline)

products = [
    "Apple iPhone smartphone",
    "Samsung Galaxy smartphone",
    "Wireless Bluetooth headphones",
    "Noise cancelling headphones",
    "Laptop computer for programming",
    "Desktop computer for gaming",
    "Smart fitness watch",
    "Digital sports smartwatch",
    "Running shoes for athletes",
    "Casual walking shoes",
    "Professional photography camera",
    "Digital camera for beginners",
    "Portable Bluetooth speaker",
    "Wireless computer mouse",
    "Mechanical computer keyboard"
]

product_embeddings = create_embeddings(products)

print("\n4. PRODUCT RECOMMENDATIONS")

product_queries = [
    "I want a mobile phone",
    "I need something for listening to music",
    "I want shoes for exercise"
]

for query in product_queries:
    print("\nCustomer request:", query)
    print("Recommended products:")
    find_similar(query, products, product_embeddings, 3)

print("\nBUSINESS USE CASE")
print("Embeddings help businesses improve search, detect duplicate complaints, organize news and recommend relevant products.")

print("\nTASK 4 COMPLETED SUCCESSFULLY!")
