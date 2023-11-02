import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)


# Apple-esque UI Enhancement
st.set_page_config(
    page_title="Apple-esque App",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Customer Purchases", "Sentiment Analysis","Inventory Analysis"])


# 1. Randomly Generate Products, Prices, and Quantities
products = ['flatscreen TVs', 'television sets', 'MP3 players', 'video recorders',
            'DVD players', 'radio receivers', 'telephones', 'cell phones',
            'e-mail-capable PCs', 'desktop computers', 'laptops', 'printers',
            'paper shredders']

prices = {
    'flatscreen TVs': np.random.randint(300, 2000),
    'television sets': np.random.randint(100, 1500),
    'MP3 players': np.random.randint(50, 400),
    'video recorders': np.random.randint(100, 800),
    'DVD players': np.random.randint(50, 200),
    'radio receivers': np.random.randint(30, 150),
    'telephones': np.random.randint(20, 150),
    'cell phones': np.random.randint(150, 1000),
    'e-mail-capable PCs': np.random.randint(500, 2000),
    'desktop computers': np.random.randint(400, 2000),
    'laptops': np.random.randint(500, 2500),
    'printers': np.random.randint(50, 500),
    'paper shredders': np.random.randint(20, 150)
}

quantities = {
    product: np.random.randint(10, 500) for product in products
}

# 2. Randomly Generate Customer Base and Purchases with One Product per Customer
customers = [f'Customer_{i}' for i in range(1000)]
locations = ['California', 'Texas', 'Florida', 'New York']
customer_purchases = {
    customer: {
        'location': np.random.choice(locations),
        'products_bought': np.random.choice(products, size=1).tolist(),
        'total_spent': sum([prices[product] for product in np.random.choice(products, size=1)])
    } for customer in customers
}

genders = ['Male', 'Female', 'Other']

for customer, data in customer_purchases.items():
    data['gender'] = np.random.choice(genders)
    data['discount'] = np.random.choice(np.arange(5, 21))  # Random discounts between 5% and 20%



customer_ratings = {
    customer: {product: np.random.randint(1,6)
               for product in purchase_data['products_bought']}
    for customer, purchase_data in customer_purchases.items()
}

df = pd.DataFrame(customer_purchases).T

# Add ratings to the DataFrame
for customer, ratings in customer_ratings.items():
    for product, rating in ratings.items():
        df.at[customer, f'rating_{product}'] = rating

cost_prices = {product: price*0.7 for product, price in prices.items()}

# Compute the total cost for all products sold
total_cost = sum([cost_prices[product] for customer_data in customer_purchases.values() for product in customer_data['products_bought']])

# Calculate total revenue (which is already calculated in the previous code)
total_revenue = df['total_spent'].sum()

# Calculate total profit
total_profit = total_revenue - total_cost


if page == "Customer Purchases":
    st.title("Customer Purchases and Ratings - Randomly Generated")
    st.dataframe(df.head(50))  # Display the first 50 rows for better layout
    st.write(f"**Total Profit:** ${total_profit:,.2f}")


    # Extract feature and target variables
    X = df[['location']]
    y = df['products_bought'].apply(lambda x: x[0])  # Since it's a list, get the product name
    le_location = LabelEncoder()
    X['location'] = le_location.fit_transform(X['location'])
    le_product = LabelEncoder()
    y = le_product.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Check accuracy on test data (optional)
    accuracy = clf.score(X_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    def predict_product(location_name):
        location_encoded = le_location.transform([location_name])
        product_encoded = clf.predict([location_encoded])
        product_name = le_product.inverse_transform(product_encoded)
        return product_name[0]

    # Visualize product distribution
    st.subheader("Product Distribution")
    product_counts = df['products_bought'].apply(lambda x: x[0]).value_counts()
    st.bar_chart(product_counts)

    # Visualize Purchases by Location
    st.subheader("Total Purchases by Location")
    location_purchases = df['location'].value_counts()
    location_df = location_purchases.reset_index()
    location_df.columns = ['Location', 'Purchases']
    chart = alt.Chart(location_df).mark_bar().encode(
        y='Location:N',
        x='Purchases:Q',
        color='Location:N',
        tooltip=['Location', 'Purchases']
    )
    st.altair_chart(chart, use_container_width=True)

    # Predict Product for a given location
    st.subheader("Product Prediction based on Location")
    location_to_predict = st.selectbox("Select a Location:", locations)
    predicted_product = predict_product(location_to_predict)
    st.write(f"The best product to sell next in {location_to_predict} is **{predicted_product}**.")

    # Display the model's accuracy
    st.write(f"Model's Accuracy: **{accuracy * 100:.2f}%**")
    
    def predict_product_priority(location_name):
        location_encoded = le_location.transform([location_name])

        # Reshape the location_encoded array to make it 2D
        location_encoded_2d = location_encoded.reshape(-1, 1)

        location_encoded_df = pd.DataFrame(location_encoded_2d, columns=['location'])

        # Use the model to predict probabilities for each product
        probabilities = clf.predict_proba(location_encoded_2d)[0]

        # Get products sorted by their probabilities
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        sorted_products = le_product.inverse_transform(sorted_indices)

        return sorted_products[:3]  # Return the top 3 products

    # Predict Product priority for a given location
    st.subheader("Product Placement Recommendation based on Location")
    location_to_predict = st.selectbox("Select a Location for Product Prediction:", locations, key='product_prediction')
    predicted_products_priority = predict_product_priority(location_to_predict)
    front_product = predicted_products_priority[0]
    second_product = predicted_products_priority[1]
    third_product = predicted_products_priority[2]

    st.write(
        f"In {location_to_predict}, **{front_product}** should be placed at the front of the store for maximum visibility.")
    st.write(
        f"Whereas, **{second_product}** and **{third_product}** can be placed farther back to optimize store layout.")

    # Display the model's accuracy
    st.write(f"Model's Accuracy: **{accuracy * 100:.2f}%**")




def simple_sentiment_analysis(review):
    positive_words = ["Loved", "Highly recommend", "Value for money", "Decent"]
    negative_words = ["Disappointed", "Not as expected"]

    for word in positive_words:
        if word in review:
            return "Positive"
    for word in negative_words:
        if word in review:
            return "Negative"
    return "Neutral"


if page == "Sentiment Analysis":
    st.title("Sentiment Analysis of Customer Reviews - Randomly Generated")

    # Generate reviews and determine sentiment
    reviews = ["Loved it!", "Not as expected.", "Decent product.", "Value for money.", "Disappointed.", "Highly recommend!"]

    # Create a DataFrame 'data' as requested
    data = {
        'Customer': [],
        'Product': [],
        'Location': [],
        'Review': [],
        'Rating': [],
        'Sentiment': []
    }

    for customer, purchase_data in customer_purchases.items():
        data['Customer'].append(customer)
        data['Product'].append(purchase_data['products_bought'][0])
        data['Location'].append(purchase_data['location'])
        review_text = np.random.choice(reviews)
        data['Review'].append(review_text)
        data['Rating'].append(np.random.randint(1, 6))
        data['Sentiment'].append(simple_sentiment_analysis(review_text))


    data_df = pd.DataFrame(data)

    st.table(data_df.head(10))  # Display the first 10 rows for better layout

    st.subheader("Sentiment Distribution")
    sentiment_counts = data_df['Sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

    # Visualize average rating by sentiment
    st.subheader("Average Rating by Sentiment")
    avg_rating = data_df.groupby('Sentiment')['Rating'].mean().sort_values(ascending=False)
    st.bar_chart(avg_rating)

    # Visualize sentiment distribution across different locations
    st.subheader("Sentiment Distribution by Location")
    location_sentiment_df = data_df.groupby(['Location', 'Sentiment']).size().unstack().fillna(0)
    st.bar_chart(location_sentiment_df)

    # Feedback mechanism on how accurate the sentiment analysis was
    feedback = st.selectbox("Was the sentiment analysis accurate?", ["Select", "Yes", "No"])
    if feedback != "Select":
        st.write("Thank you for your feedback!")

# Inventory Analysis
if page == "Inventory Analysis":
    st.title("Inventory Analysis and Prediction - Randomly Generated")

    # Generate Inventory Data
    dates = pd.date_range(start="2023-11-13", end="2023-11-20", freq='D')
    # ... [rest of the inventory data generation code]

    dates = pd.date_range(start="2023-11-13", end="2023-11-20", freq='D')
    data = []

    for date in dates:
        for product in products:
            data.append({
                'date': date,
                'product': product,
                'price': prices[product],
                'available': np.random.randint(1, quantities[product]),
                'sold': np.random.randint(1, 100)
            })

    inventory_df = pd.DataFrame(data)
    st.dataframe(inventory_df.head(50))

    predictions = {}
    for product in products:
        product_data = inventory_df[inventory_df['product'] == product]
        X = np.array(range(len(product_data))).reshape(-1, 1)
        y = product_data['sold'].values
        model = LinearRegression().fit(X, y)
        predictions[product] = model.predict(X)

    # Add the predictions to the inventory_df
    inventory_df['prediction'] = np.concatenate(list(predictions.values()))

    # ... [code where the prediction is added to inventory_df]

    # Inventory Prediction Chart
    st.subheader("Product Sales Prediction")
    selected_product = st.selectbox("Select a product:", products)
    product_data = inventory_df[inventory_df['product'] == selected_product]

    # Altair chart for Actual Sales
    actual_chart = alt.Chart(product_data).mark_line(color='blue').encode(
        x='date:T',
        y='sold:Q',
        tooltip=['date:T', 'sold:Q']
    ).properties(
        title='Actual Sales'
    )

    # Altair chart for Predicted Sales
    predicted_chart = alt.Chart(product_data).mark_line(color='red').encode(
        x='date:T',
        y='prediction:Q',
        tooltip=['date:T', 'prediction:Q']
    ).properties(
        title='Predicted Sales'
    )

    # Altair combined chart
    combined_chart = actual_chart + predicted_chart

    # Display the charts in Streamlit
    st.altair_chart(actual_chart, use_container_width=True)
    st.altair_chart(predicted_chart, use_container_width=True)
    st.altair_chart(combined_chart, use_container_width=True)

    # Inventory Recommendations
    st.subheader("Inventory Recommendations")
    predicted_sales = product_data['prediction'].sum()
    actual_sales = product_data['sold'].sum()

    st.subheader("Inventory Recommendations")
    avg_predicted = inventory_df.groupby('product')['prediction'].mean()
    avg_sold = inventory_df.groupby('product')['sold'].mean()

    more_demand = avg_predicted[avg_predicted > avg_sold].count()
    less_demand = avg_predicted[avg_predicted <= avg_sold].count()

    pie_df = pd.DataFrame({
        'Recommendation': ['Stock More', 'Stock Less'],
        'Count': [more_demand, less_demand]
    })

    pie_chart = alt.Chart(pie_df).mark_arc(innerRadius=50).encode(
        theta='Count:Q',
        color=alt.Color('Recommendation:N',
                        scale=alt.Scale(domain=['Stock More', 'Stock Less'], range=['#3498DB', '#E74C3C'])),
        tooltip=['Recommendation', 'Count']
    ).properties(
        title='Inventory Recommendations',
        width=300,
        height=300
    )

    st.altair_chart(pie_chart, use_container_width=True)

    if predicted_sales > actual_sales:
        st.success(f"📈 **{selected_product}** is predicted to have an increase in sales. Consider stocking more!")
    else:
        st.warning(f"📉 **{selected_product}** is predicted to have a decrease in sales. Consider reducing inventory!")

        # Overall product recommendations with pie chart
    avg_predicted = inventory_df.groupby('product')['prediction'].mean()
    avg_sold = inventory_df.groupby('product')['sold'].mean()

    more_demand = avg_predicted[avg_predicted > avg_sold].index.tolist()
    less_demand = avg_predicted[avg_predicted <= avg_sold].index.tolist()

    pie_df = pd.DataFrame({
        'Recommendation': ['Stock More'] * len(more_demand) + ['Stock Less'] * len(less_demand),
        'Products': more_demand + less_demand
    })

    pie_chart = alt.Chart(pie_df).mark_bar().encode(
        x='Recommendation',
        y=alt.Y('count():Q', title="Number of Products"),
        color=alt.Color('Recommendation:N',
                        scale=alt.Scale(domain=['Stock More', 'Stock Less'], range=['#3498DB', '#E74C3C'])),
        tooltip=['Recommendation', 'Products']
    ).properties(title="Inventory Recommendations Overview")

    st.altair_chart(pie_chart, use_container_width=True)

    st.write(f"📈 Products predicted to sell more and might require more inventory: {', '.join(more_demand)}")
    st.write(f"📉 Products predicted to sell less and might require less inventory: {', '.join(less_demand)}")

    # Bar chart for avg predicted vs avg sold
    bar_df = pd.DataFrame({
        'Product': products,
        'Average Predicted Sales': avg_predicted.tolist(),
        'Average Actual Sales': avg_sold.tolist()
    })

    bar_chart = alt.Chart(bar_df).mark_bar().encode(
        x='Product',
        y='Average Predicted Sales',
        y2='Average Actual Sales',
        color=alt.condition(
            alt.datum['Average Predicted Sales'] > alt.datum['Average Actual Sales'],
            alt.value("#3498DB"),  # blue: predicted > actual
            alt.value("#E74C3C")  # red: predicted < actual
        ),
        tooltip=['Product', 'Average Predicted Sales', 'Average Actual Sales']
    ).properties(title="Predicted vs Actual Sales")

    st.altair_chart(bar_chart, use_container_width=True)




