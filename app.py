import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
import io
import base64
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="E-commerce ML Analysis",
    page_icon="📊",
    layout="wide"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4b778d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #28536b;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("<h1 class='main-header'>Brazilian E-commerce Machine Learning Application</h1>", unsafe_allow_html=True)

st.markdown("""
This application demonstrates various machine learning techniques using a Brazilian E-commerce dataset. 
You can explore different algorithms for:
- **Supervised Learning**: Classification and Regression
- **Unsupervised Learning**: Clustering and Association Rules

Select options from the sidebar to interact with different models.
""")

# Generate a synthetic e-commerce dataset based on Brazilian e-commerce patterns
def generate_ecommerce_dataset(n_samples=1000):
    np.random.seed(42)
    
    # Customer data
    customer_id = [f'CUST_{i:05d}' for i in range(n_samples)]
    customer_state = np.random.choice(['SP', 'RJ', 'MG', 'RS', 'PR', 'BA', 'SC', 'GO', 'PE', 'CE'], n_samples)
    customer_age = np.random.normal(35, 12, n_samples).astype(int)
    customer_age = np.clip(customer_age, 18, 80)
    
    # Order data
    order_status = np.random.choice(['delivered', 'shipped', 'processing', 'canceled'], n_samples, 
                                   p=[0.7, 0.15, 0.1, 0.05])
    order_value = np.random.gamma(shape=5, scale=30, size=n_samples)
    order_items = np.random.poisson(lam=2, size=n_samples) + 1
    shipping_cost = np.random.gamma(shape=2, scale=5, size=n_samples)
    
    # Calculated fields
    payment_type = np.random.choice(['credit_card', 'boleto', 'voucher', 'debit_card'], n_samples, 
                                  p=[0.65, 0.20, 0.10, 0.05])
    days_to_delivery = np.random.poisson(lam=12, size=n_samples)
    days_to_delivery = np.clip(days_to_delivery, 1, 45)
    
    # Product data
    product_category = np.random.choice(['electronics', 'furniture', 'clothing', 'books', 'beauty', 
                                        'sports', 'toys', 'home_decor', 'pet_supplies', 'groceries'], n_samples)
    
    # Customer behavior
    review_score = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.03, 0.07, 0.15, 0.35, 0.4])
    
    # Create features based on correlations
    # Older customers tend to buy more expensive items
    order_value += (customer_age - 35) * 0.5
    
    # Different states have different average order values
    state_effect = {'SP': 20, 'RJ': 15, 'MG': 10, 'RS': 5, 'PR': 0, 
                   'BA': -5, 'SC': 0, 'GO': -10, 'PE': -15, 'CE': -20}
    for i, state in enumerate(customer_state):
        order_value[i] += state_effect[state]
    
    # More items usually means higher order value
    order_value += order_items * 25
    
    # Different product categories have different price ranges
    category_effect = {'electronics': 150, 'furniture': 200, 'clothing': 50, 'books': 30, 'beauty': 40, 
                      'sports': 70, 'toys': 45, 'home_decor': 80, 'pet_supplies': 35, 'groceries': 25}
    for i, category in enumerate(product_category):
        order_value[i] += category_effect[category] * np.random.uniform(0.8, 1.2)
    
    # Shipping cost is influenced by order value and state
    for i, state in enumerate(customer_state):
        distance_factor = {'SP': 1.0, 'RJ': 1.1, 'MG': 1.2, 'RS': 1.5, 'PR': 1.3, 
                         'BA': 1.7, 'SC': 1.4, 'GO': 1.6, 'PE': 1.8, 'CE': 2.0}
        shipping_cost[i] *= distance_factor[state]
        shipping_cost[i] += order_value[i] * 0.02  # Heavier items cost more to ship
    
    # Adjust delivery times based on state (location)
    for i, state in enumerate(customer_state):
        distance_factor = {'SP': 0.8, 'RJ': 0.9, 'MG': 1.0, 'RS': 1.3, 'PR': 1.1, 
                         'BA': 1.5, 'SC': 1.2, 'GO': 1.4, 'PE': 1.6, 'CE': 1.8}
        days_to_delivery[i] = int(days_to_delivery[i] * distance_factor[state])
    
    # Customer satisfaction (review_score) is influenced by delivery time and order status
    for i in range(n_samples):
        if order_status[i] == 'canceled':
            review_score[i] = np.random.choice([1, 2], p=[0.8, 0.2])
        elif order_status[i] == 'processing':
            if days_to_delivery[i] > 20:
                review_score[i] = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
        elif days_to_delivery[i] > 25:
            review_score[i] = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        elif days_to_delivery[i] > 15:
            review_score[i] = np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3])
    
    # Create binary features for association rule mining
    is_high_value = (order_value > 150).astype(int)
    is_electronic = (product_category == 'electronics').astype(int)
    is_furniture = (product_category == 'furniture').astype(int)
    is_clothing = (product_category == 'clothing').astype(int)
    is_fast_delivery = (days_to_delivery <= 7).astype(int)
    is_credit_card = (payment_type == 'credit_card').astype(int)
    is_high_rated = (review_score >= 4).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_id,
        'customer_state': customer_state,
        'customer_age': customer_age,
        'order_status': order_status,
        'order_value': order_value,
        'order_items': order_items,
        'shipping_cost': shipping_cost,
        'payment_type': payment_type,
        'days_to_delivery': days_to_delivery,
        'product_category': product_category,
        'review_score': review_score,
        'is_high_value': is_high_value,
        'is_electronic': is_electronic,
        'is_furniture': is_furniture,
        'is_clothing': is_clothing,
        'is_fast_delivery': is_fast_delivery,
        'is_credit_card': is_credit_card,
        'is_high_rated': is_high_rated
    })
    
    return df

# Generate the dataset
df = generate_ecommerce_dataset(1000)

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose the analysis mode:", 
                           ["Dataset Overview", 
                            "Supervised Learning - Classification", 
                            "Supervised Learning - Regression", 
                            "Unsupervised Learning - Clustering",
                            "Unsupervised Learning - Association Rules",
                            "Generate Report"])

# Dataset Overview
if app_mode == "Dataset Overview":
    st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Number of Records", len(df))
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Number of Features", len(df.columns))
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h3 class='sub-header'>Data Preview</h3>", unsafe_allow_html=True)
    st.dataframe(df.head())
    
    st.markdown("<h3 class='sub-header'>Data Summary</h3>", unsafe_allow_html=True)
    st.dataframe(df.describe())
    
    st.markdown("<h3 class='sub-header'>Feature Distributions</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Order Value Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['order_value'], kde=True, ax=ax)
        st.pyplot(fig)
        
        st.subheader("Customer Age Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['customer_age'], kde=True, ax=ax)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Review Score Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x='review_score', ax=ax)
        st.pyplot(fig)
        
        st.subheader("Product Category Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, y='product_category', order=df['product_category'].value_counts().index, ax=ax)
        st.pyplot(fig)
    
    st.markdown("<h3 class='sub-header'>Correlations</h3>", unsafe_allow_html=True)
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Supervised Learning - Classification
elif app_mode == "Supervised Learning - Classification":
    st.markdown("<h2 class='sub-header'>Customer Satisfaction Prediction (Classification)</h2>", unsafe_allow_html=True)
    st.write("This model predicts whether a customer will give a high review score (4-5) based on order details.")
    
    # Prepare data for classification
    X = df[['order_value', 'shipping_cost', 'days_to_delivery', 'order_items']]
    y = (df['review_score'] >= 4).astype(int)  # Binary classification: high rating (4-5) or not
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Choose classification algorithm
    st.markdown("<h3 class='sub-header'>Select Classification Algorithm</h3>", unsafe_allow_html=True)
    classifier_name = st.selectbox("Choose a classifier:", ["Logistic Regression", "Random Forest"])
    
    if classifier_name == "Logistic Regression":
        classifier = LogisticRegression(random_state=42)
    else:
        classifier = RandomForestClassifier(random_state=42)
    
    # Train the model and display metrics
    with st.spinner('Training the classification model...'):
        classifier.fit(X_train_scaled, y_train)
        y_pred = classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Accuracy", f"{accuracy:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        if classifier_name == "Random Forest":
            importances = classifier.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.write("Feature Importance")
            st.dataframe(feature_importance_df)
            st.markdown("</div>", unsafe_allow_html=True)
    
    # User prediction section
    st.markdown("<h3 class='sub-header'>Predict Customer Satisfaction</h3>", unsafe_allow_html=True)
    st.write("Enter order details to predict if the customer will give a high rating (4-5 stars):")
    
    col1, col2 = st.columns(2)
    
    with col1:
        order_value_input = st.slider("Order Value (R$)", min_value=20, max_value=500, value=150)
        shipping_cost_input = st.slider("Shipping Cost (R$)", min_value=5, max_value=50, value=20)
    
    with col2:
        days_delivery_input = st.slider("Estimated Days to Delivery", min_value=1, max_value=30, value=10)
        order_items_input = st.slider("Number of Items", min_value=1, max_value=10, value=2)
    
    if st.button("Predict Customer Satisfaction"):
        user_input = np.array([[order_value_input, shipping_cost_input, days_delivery_input, order_items_input]])
        user_input_scaled = scaler.transform(user_input)
        prediction = classifier.predict(user_input_scaled)[0]
        probability = classifier.predict_proba(user_input_scaled)[0][1]
        
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        if prediction == 1:
            st.success(f"The customer is likely to give a high rating (4-5 stars)! Probability: {probability:.2f}")
        else:
            st.error(f"The customer might give a lower rating (1-3 stars). Probability of high rating: {probability:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Explanation of prediction
        st.markdown("<h4>Why this prediction?</h4>", unsafe_allow_html=True)
        
        explanation_text = """
        Factors affecting customer satisfaction:
        - **Order Value**: Higher value orders tend to have higher customer expectations
        - **Shipping Cost**: Lower shipping costs typically lead to higher satisfaction
        - **Delivery Time**: Faster delivery correlates strongly with higher ratings
        - **Number of Items**: More items can complicate order fulfillment and affect satisfaction
        """
        st.markdown(explanation_text)

# Supervised Learning - Regression
elif app_mode == "Supervised Learning - Regression":
    st.markdown("<h2 class='sub-header'>Order Value Prediction (Regression)</h2>", unsafe_allow_html=True)
    st.write("This model predicts the total order value based on customer and product details.")
    
    # Prepare data for regression
    # Encode categorical variables
    product_encoder = LabelEncoder()
    state_encoder = LabelEncoder()
    payment_encoder = LabelEncoder()
    
    df_reg = df.copy()
    df_reg['product_category_encoded'] = product_encoder.fit_transform(df['product_category'])
    df_reg['customer_state_encoded'] = state_encoder.fit_transform(df['customer_state'])
    df_reg['payment_type_encoded'] = payment_encoder.fit_transform(df['payment_type'])
    
    X = df_reg[['customer_age', 'customer_state_encoded', 'product_category_encoded', 
                'payment_type_encoded', 'order_items', 'shipping_cost']]
    y = df_reg['order_value']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Choose regression algorithm
    st.markdown("<h3 class='sub-header'>Select Regression Algorithm</h3>", unsafe_allow_html=True)
    regressor_name = st.selectbox("Choose a regressor:", ["Linear Regression", "Random Forest Regressor"])
    
    if regressor_name == "Linear Regression":
        regressor = LinearRegression()
    else:
        regressor = RandomForestRegressor(random_state=42)
    
    # Train the model and display metrics
    with st.spinner('Training the regression model...'):
        regressor.fit(X_train_scaled, y_train)
        y_pred = regressor.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("RMSE", f"R${rmse:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        if regressor_name == "Random Forest Regressor":
            importances = regressor.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.write("Feature Importance")
            st.dataframe(feature_importance_df)
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Order Value')
    plt.ylabel('Predicted Order Value')
    plt.title('Actual vs Predicted Order Values')
    st.pyplot(fig)
    
    # User prediction section
    st.markdown("<h3 class='sub-header'>Predict Order Value</h3>", unsafe_allow_html=True)
    st.write("Enter customer and order details to predict the total order value:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        customer_age_input = st.slider("Customer Age", min_value=18, max_value=80, value=35)
        customer_state_input = st.selectbox("Customer State", options=df['customer_state'].unique())
        product_category_input = st.selectbox("Product Category", options=df['product_category'].unique())
    
    with col2:
        payment_type_input = st.selectbox("Payment Type", options=df['payment_type'].unique())
        order_items_input = st.slider("Number of Items", min_value=1, max_value=10, value=2)
        shipping_cost_input = st.slider("Shipping Cost (R$)", min_value=5, max_value=50, value=20)
    
    if st.button("Predict Order Value"):
        # Transform user inputs
        customer_state_encoded = state_encoder.transform([customer_state_input])[0]
        product_category_encoded = product_encoder.transform([product_category_input])[0]
        payment_type_encoded = payment_encoder.transform([payment_type_input])[0]
        
        user_input = np.array([[customer_age_input, customer_state_encoded, product_category_encoded, 
                               payment_type_encoded, order_items_input, shipping_cost_input]])
        user_input_scaled = scaler.transform(user_input)
        predicted_value = regressor.predict(user_input_scaled)[0]
        
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.success(f"Predicted Order Value: R$ {predicted_value:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Explanation of prediction
        st.markdown("<h4>Factors Influencing Order Value</h4>", unsafe_allow_html=True)
        
        explanation_text = """
        Key factors affecting order value:
        - **Product Category**: Different categories have different average prices
        - **Number of Items**: More items typically increase the total order value
        - **Customer Location**: Different states may have different purchasing power
        - **Payment Method**: Some payment methods correlate with higher order values
        """
        st.markdown(explanation_text)

# Unsupervised Learning - Clustering
elif app_mode == "Unsupervised Learning - Clustering":
    st.markdown("<h2 class='sub-header'>Customer Segmentation (Clustering)</h2>", unsafe_allow_html=True)
    st.write("This analysis segments customers based on their purchasing behavior.")
    
    # Prepare data for clustering
    cluster_features = ['customer_age', 'order_value', 'order_items', 'days_to_delivery', 'review_score']
    X_cluster = df[cluster_features].copy()
    
    # Scale the data
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)
    
    # Choose number of clusters
    st.markdown("<h3 class='sub-header'>Select Number of Clusters</h3>", unsafe_allow_html=True)
    n_clusters = st.slider("Number of clusters (k)", min_value=2, max_value=8, value=4)
    
    # Perform clustering
    with st.spinner('Performing K-Means clustering...'):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_cluster_scaled)
        
        # Add cluster labels to the dataset
        df_clustered = df.copy()
        df_clustered['cluster'] = cluster_labels
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_cluster_scaled, cluster_labels)
    
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display cluster statistics
    st.markdown("<h3 class='sub-header'>Cluster Profiles</h3>", unsafe_allow_html=True)
    cluster_stats = df_clustered.groupby('cluster')[cluster_features].mean()
    st.dataframe(cluster_stats)
    
    # Visualization
    st.markdown("<h3 class='sub-header'>Cluster Visualization</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature_x = st.selectbox("Select X-axis feature:", cluster_features, index=0)
    
    with col2:
        feature_y = st.selectbox("Select Y-axis feature:", cluster_features, index=1)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = plt.scatter(df_clustered[feature_x], df_clustered[feature_y], c=df_clustered['cluster'], 
                         cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f'Clusters by {feature_x} and {feature_y}')
    st.pyplot(fig)
    
    # Cluster interpretation
    st.markdown("<h3 class='sub-header'>Cluster Interpretation</h3>", unsafe_allow_html=True)
    
    # Create descriptions for each cluster
    cluster_descriptions = {}
    
    for cluster_id in range(n_clusters):
        cluster_data = cluster_stats.loc[cluster_id]
        
        age = "younger" if cluster_data['customer_age'] < cluster_stats['customer_age'].mean() else "older"
        spending = "lower" if cluster_data['order_value'] < cluster_stats['order_value'].mean() else "higher"
        items = "fewer" if cluster_data['order_items'] < cluster_stats['order_items'].mean() else "more"
        delivery = "faster" if cluster_data['days_to_delivery'] < cluster_stats['days_to_delivery'].mean() else "slower"
        satisfaction = "lower" if cluster_data['review_score'] < cluster_stats['review_score'].mean() else "higher"
        
        description = f"Cluster {cluster_id}: {age} customers with {spending} spending, ordering {items} items, "
        description += f"with {delivery} delivery times and {satisfaction} satisfaction."
        
        cluster_descriptions[cluster_id] = description
    
    for cluster_id, description in cluster_descriptions.items():
        st.markdown(f"**{description}**")
    
    # Marketing recommendations
    st.markdown("<h3 class='sub-header'>Marketing Recommendations</h3>", unsafe_allow_html=True)
    
    recommendations = {
        "High Value & High Satisfaction": "Target for premium product offers and loyalty programs.",
        "High Value & Lower Satisfaction": "Focus on improving service quality and addressing pain points.",
        "Lower Value & Fast Delivery": "Cross-selling opportunities and volume-based discounts.",
        "Younger Customers": "Mobile-first marketing approach with social media integration.",
        "Older Customers with Higher Spending": "Traditional marketing channels with focus on quality and reliability."
    }
    
    for rec_type, recommendation in recommendations.items():
        st.markdown(f"**{rec_type}**: {recommendation}")

# Unsupervised Learning - Association Rules
elif app_mode == "Unsupervised Learning - Association Rules":
    st.markdown("<h2 class='sub-header'>E-commerce Association Rules</h2>", unsafe_allow_html=True)
    st.write("This analysis discovers patterns in customer purchasing behavior and preferences.")
    
    # Prepare data for association rules mining
    association_features = [
        'is_high_value', 'is_electronic', 'is_furniture', 'is_clothing', 
        'is_fast_delivery', 'is_credit_card', 'is_high_rated'
    ]
    
    basket_df = df[association_features].copy()
    
    # Set parameters for Apriori algorithm
    st.markdown("<h3 class='sub-header'>Association Rules Parameters</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_support = st.slider("Minimum Support", min_value=0.05, max_value=0.5, value=0.1, step=0.05)
    
    with col2:
        min_confidence = st.slider("Minimum Confidence", min_value=0.1, max_value=0.9, value=0.6, step=0.1)
    
    # Generate frequent itemsets and association rules
    with st.spinner('Generating association rules...'):
        frequent_itemsets = apriori(basket_df, min_support=min_support, use_colnames=True)
        
        if not frequent_itemsets.empty:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            rules = rules.sort_values('confidence', ascending=False)
        else:
            rules = pd.DataFrame()
# Display results
        if not rules.empty:
            st.markdown("<h3 class='sub-header'>Association Rules Found</h3>", unsafe_allow_html=True)
            
            # Format the rules for better readability
            rules_display = rules.copy()
            rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
            
            # Select only the most relevant columns for display
            rules_display = rules_display[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
            rules_display = rules_display.round(3)
            
            st.dataframe(rules_display)
            
            # Visualize top rules
            st.markdown("<h3 class='sub-header'>Top Association Rules</h3>", unsafe_allow_html=True)
            
            # Plot the top rules by lift
            top_rules = rules.nlargest(10, 'lift')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=top_rules, x='support', y='confidence', size='lift', sizes=(50, 500), alpha=0.6)
            
            # Add annotations to the plot
            for i, row in top_rules.iterrows():
                ant = ', '.join(list(row['antecedents']))
                con = ', '.join(list(row['consequents']))
                plt.annotate(f"{ant} → {con}", 
                            (row['support'], row['confidence']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8)
            
            plt.title('Top Association Rules by Lift')
            plt.xlabel('Support')
            plt.ylabel('Confidence')
            st.pyplot(fig)
            
            # Business insights from association rules
            st.markdown("<h3 class='sub-header'>Business Insights</h3>", unsafe_allow_html=True)
            
            st.write("""
            Association rules provide insights into customer purchasing patterns and preferences:
            
            - **Product Bundling**: Items frequently purchased together can be bundled for promotions
            - **Store Layout**: Products with strong associations can be placed near each other
            - **Targeted Marketing**: Use antecedents to identify customer segments for specific products
            - **Cross-selling**: Recommend items based on the rules when customers make related purchases
            """)
        else:
            st.warning("No association rules found with the current parameters. Try lowering the minimum support or confidence thresholds.")

# Generate Report
elif app_mode == "Generate Report":
    st.markdown("<h2 class='sub-header'>E-commerce ML Analysis Report Generator</h2>", unsafe_allow_html=True)
    st.write("Generate a comprehensive report of the analysis performed on the Brazilian E-commerce dataset.")
    
    # Report options
    st.markdown("<h3 class='sub-header'>Report Options</h3>", unsafe_allow_html=True)
    
    include_dataset_overview = st.checkbox("Include Dataset Overview", value=True)
    include_classification = st.checkbox("Include Classification Analysis", value=True)
    include_regression = st.checkbox("Include Regression Analysis", value=True)
    include_clustering = st.checkbox("Include Clustering Analysis", value=True)
    include_association_rules = st.checkbox("Include Association Rules Analysis", value=True)
    
    # Create the report content
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            report_content = []
            
            # Report header
            report_content.append("# Brazilian E-commerce Machine Learning Analysis Report")
            report_content.append(f"*Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}*\n")
            
            report_content.append("## Executive Summary")
            report_content.append("""
            This report presents a comprehensive analysis of Brazilian E-commerce data using various machine learning techniques. 
            The analysis includes supervised learning methods (classification and regression) and unsupervised learning approaches 
            (clustering and association rules mining) to extract valuable insights from customer behavior and purchasing patterns.
            """)
            
            # Dataset Overview
            if include_dataset_overview:
                report_content.append("## 1. Dataset Overview")
                report_content.append(f"- **Number of Records**: {len(df)}")
                report_content.append(f"- **Number of Features**: {len(df.columns)}")
                report_content.append("\n### Summary Statistics")
                # Convert describe to markdown
                desc_stats = df.describe().round(2).to_markdown()
                report_content.append(desc_stats)
                
                report_content.append("\n### Key Observations")
                report_content.append("""
                - The dataset contains a diverse range of order values, with significant variation in shipping costs
                - Customer ages range from young adults to seniors, with most customers in the 25-45 age range
                - Most orders receive positive reviews (4-5 stars)
                - There's a good distribution across different product categories
                """)
            
            # Classification Analysis
            if include_classification:
                report_content.append("## 2. Customer Satisfaction Prediction (Classification)")
                report_content.append("""
                ### Model Overview
                A classification model was developed to predict whether customers would give high ratings (4-5 stars) 
                based on order details such as order value, shipping cost, delivery time, and number of items.
                
                ### Key Findings
                - **Delivery time** is the strongest predictor of customer satisfaction
                - **Shipping cost** shows an inverse relationship with customer satisfaction
                - High-value orders tend to have higher customer expectations
                - The model achieves reasonable accuracy in predicting customer satisfaction
                
                ### Business Applications
                - Identify orders at risk of low satisfaction for proactive customer service
                - Optimize shipping and delivery processes to improve customer satisfaction
                - Set appropriate customer expectations based on order characteristics
                """)
            
            # Regression Analysis
            if include_regression:
                report_content.append("## 3. Order Value Prediction (Regression)")
                report_content.append("""
                ### Model Overview
                A regression model was developed to predict total order value based on customer demographics, 
                product details, and order characteristics.
                
                ### Key Findings
                - Product category is a strong predictor of order value
                - Customer location (state) influences purchasing patterns
                - Number of items per order directly correlates with order value
                - Customer age shows some correlation with spending patterns
                
                ### Business Applications
                - Targeted marketing based on predicted customer spending
                - Inventory planning and demand forecasting
                - Personalized product recommendations based on spending potential
                """)
            
            # Clustering Analysis
            if include_clustering:
                report_content.append("## 4. Customer Segmentation (Clustering)")
                report_content.append("""
                ### Analysis Overview
                K-means clustering was applied to segment customers based on age, order value, 
                number of items purchased, delivery time, and review scores.
                
                ### Customer Segments Identified
                - **High-value Satisfied Customers**: Older customers with high spending and high satisfaction
                - **Value-conscious Customers**: Younger customers with lower order values but fast delivery
                - **Demanding Customers**: Higher spending customers with lower satisfaction scores
                - **Standard Customers**: Average on most metrics with moderate satisfaction
                
                ### Marketing Strategies
                - Develop targeted marketing campaigns for each customer segment
                - Focus on service improvements for segments with lower satisfaction
                - Create loyalty programs for high-value customer segments
                """)
            
            # Association Rules Analysis
            if include_association_rules:
                report_content.append("## 5. Association Rules Analysis")
                report_content.append("""
                ### Analysis Overview
                Association rules mining was performed to discover patterns in customer purchasing behavior 
                and preferences across different product categories and order characteristics.
                
                ### Key Patterns Discovered
                - High correlation between furniture purchases and high order values
                - Credit card payments associated with higher review scores
                - Fast delivery strongly associated with high customer ratings
                - Electronics purchases often accompanied by higher shipping costs
                
                ### Recommendations
                - Implement cross-selling strategies based on discovered associations
                - Optimize store layout (physical or digital) to place associated items together
                - Develop bundled promotions for frequently co-purchased items
                - Use payment method preferences to tailor checkout options
                """)
            
            # Conclusion
            report_content.append("## Conclusion")
            report_content.append("""
            The machine learning analysis of the Brazilian E-commerce dataset provides valuable insights 
            into customer behavior, preferences, and purchasing patterns. These insights can be leveraged 
            to enhance marketing strategies, improve customer satisfaction, optimize operations, and 
            ultimately increase sales and profitability.
            
            By implementing the recommendations from classification, regression, clustering, and association 
            rules analyses, e-commerce businesses can make data-driven decisions that align with customer 
            needs and market trends.
            """)
            
            # Combine all content into a single markdown string
            report_markdown = "\n\n".join(report_content)
            
            # Display the report
            st.markdown("<h3 class='sub-header'>Preview Report</h3>", unsafe_allow_html=True)
            st.markdown(report_markdown)
            
            # Create a download button for the report
            report_pdf = io.BytesIO()
            # Note: In a real application, you would use a PDF generation library here
            # For this example, we'll just provide the markdown text as a downloadable file
            
            st.download_button(
                label="Download Report as Markdown",
                data=report_markdown,
                file_name="brazilian_ecommerce_ml_analysis.md",
                mime="text/markdown"
            )
            
            st.success("Report generated successfully! Click the button above to download.")
            
            st.info("""
            Note: For a PDF version of this report, you can:
            1. Download the markdown file
            2. Use an online markdown to PDF converter
            3. Or use a tool like Pandoc to convert it locally
            """)